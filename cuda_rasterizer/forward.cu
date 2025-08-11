/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx,//高斯点索引
	 									int deg,//球谐函数的阶数
										int max_coeffs,//每个高斯的球谐函数系数个数
										const glm::vec3* means,
										glm::vec3 campos,//相机位置
										const float* shs,//球谐系数数组(所有高斯的球谐系数展平)
										bool* clamped)
{
	// The implementation is loosely based on code for
	// "Differentiable Point-Based Radiance Fields for
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];//点坐标
	glm::vec3 dir = pos - campos;//相机->点的向量
	dir = dir / glm::length(dir);//相机到目标点之间的距离

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;//每个高斯的球谐系数指针指向第一个球谐系数
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Compute a 2D-to-2D mapping matrix from a tangent plane into a image plane
// given a 2D gaussian parameters.
__device__ bool computeTransMat(const glm::vec3 &p_world, const glm::vec4 &quat, const glm::vec2 &scale, const float *viewmat, const float4 &intrins, float tan_fovx, float tan_fovy, float* transMat, float3 &normal) {
	// Setup cameras
	// Currently only support ideal pinhole camera
	// but more advanced intrins can be implemented
	const glm::mat3 W = glm::mat3(
		viewmat[0],viewmat[1],viewmat[2],
		viewmat[4],viewmat[5],viewmat[6],
		viewmat[8],viewmat[9],viewmat[10]
	);//旋转矩阵
	const glm::vec3 cam_pos = glm::vec3(viewmat[12], viewmat[13], viewmat[14]); // camera center
	//平移矩阵
	const glm::mat4 P = glm::mat4(
		intrins.x, 0.0, 0.0, 0.0,
		0.0, intrins.y, 0.0, 0.0,
		intrins.z, intrins.w, 1.0, 1.0,
		0.0, 0.0, 0.0, 0.0
	);

	// Make the geometry of 2D Gaussian as a Homogeneous transformation matrix
	// under the camera view, See Eq. (5) in 2DGS' paper.
	glm::vec3 p_view = W * p_world + cam_pos;//点的世界坐标转为相机坐标

	//quat和scale是高斯的旋转和缩放
	glm::mat3 R = quat_to_rotmat(quat) * scale_to_mat({scale.x, scale.y, 1.0f}, 1.0f);
	glm::mat3 M = glm::mat3(W * R[0], W * R[1], p_view);//表示在相机坐标系中高斯的旋转、缩放和位置信息
	glm::vec3 tn = W*R[2];
	float cos = glm::dot(-tn, p_view);//高斯的朝向和相机看向高斯的方向之间的夹角

#if BACKFACE_CULL
	if (cos == 0.0f) return false;//不可见
#endif

#if RENDER_AXUTILITY and DUAL_VISIABLE
	// This means a 2D Gaussian is dual visiable.
	// Experimentally, turning off the dual visiable works eqully.
	float multiplier = cos > 0 ? 1 : -1;
	tn *= multiplier;//将法线调整为面向相机的方向
#endif
	// projection into screen space, see Eq. (7)
	glm::mat4x3 T = glm::transpose(P * glm::mat3x4(
		glm::vec4(M[0], 0.0),
		glm::vec4(M[1], 0.0),
		glm::vec4(M[2], 1.0)
	));

	transMat[0] = T[0].x;
	transMat[1] = T[0].y;
	transMat[2] = T[0].z;
	transMat[3] = T[1].x;
	transMat[4] = T[1].y;
	transMat[5] = T[1].z;
	transMat[6] = T[2].x;
	transMat[7] = T[2].y;
	transMat[8] = T[2].z;
	normal = {tn.x, tn.y, tn.z};//摄像机坐标系下的高斯法线方向
	return true;
}

// Computing the bounding box of the 2D Gaussian and its center,
// where the center of the bounding box is used to create a low pass filter
// in the image plane
__device__ bool computeAABB(const float *transMat, float2 & center, float2 & extent, int W, int H) {
	glm::mat4x3 T = glm::mat4x3(
		transMat[0], transMat[1], transMat[2],
		transMat[3], transMat[4], transMat[5],
		transMat[6], transMat[7], transMat[8],
		transMat[6], transMat[7], transMat[8]
	);

	float d = glm::dot(glm::vec3(1.0, 1.0, -1.0), T[3] * T[3]);

	if (d == 0.0f) return false;

	glm::vec3 f = glm::vec3(1.0, 1.0, -1.0) * (1.0f / d);

	glm::vec3 p = glm::vec3(
		glm::dot(f, T[0] * T[3]),
		glm::dot(f, T[1] * T[3]),
		glm::dot(f, T[2] * T[3]));
	
	if (p.x < -W/4 || p.x > W*5/4 || p.y < -H/4 || p.y > H*5/4)
		return false;

	glm::vec3 h0 = p * p -
		glm::vec3(
			glm::dot(f, T[0] * T[0]),
			glm::dot(f, T[1] * T[1]),
			glm::dot(f, T[2] * T[2])
		);

	glm::vec3 h = sqrt(max(glm::vec3(0.0), h0)) + glm::vec3(0.0, 0.0, 1e-2);
	center = {p.x, p.y};
	extent = {h.x, h.y};
	return true;
}

//依据夹角计算权重
__device__ bool compute_weight(const glm::vec3 &p_world, const glm::vec4 &quat, const glm::vec2 &scale, const float *viewmat, float &weight)
//点坐标和外参用于计算光线方向，旋转和缩放矩阵用于计算高斯朝向
{
	//构建相机到高斯的光线
	const glm::mat3 W = glm::mat3(
		viewmat[0],viewmat[1],viewmat[2],
		viewmat[4],viewmat[5],viewmat[6],
		viewmat[8],viewmat[9],viewmat[10]
	);//旋转矩阵
	const glm::vec3 cam_pos = glm::vec3(viewmat[12], viewmat[13], viewmat[14]); // camera center
	//平移矩阵

	glm::vec3 p_view = W * p_world + cam_pos;//点的世界坐标转为相机坐标
	//quat和scale是高斯的旋转和缩放

	//构建高斯的法线
	glm::mat3 R = quat_to_rotmat(quat) * scale_to_mat({scale.x, scale.y, 1.0f}, 1.0f);
	glm::vec3 tn = W*R[2];
	float cos = glm::dot(-tn, p_view);//高斯的朝向和相机看向高斯的方向之间的夹角
	weight=abs(cos);//构建权重（根据不同的公式）
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec2* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* transMat_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, const float tan_fovy,
	const float focal_x, const float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* transMats,
	float* rgb,
	float4* normal_opacity,
	float* weights,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	auto idx = cg::this_grid().thread_rank();
	//线程序号就是处理点的序号
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	glm::vec3 p_world = glm::vec3(orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2]);//点的世界坐标
	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
	//判断点是否在视锥体里
		return;

	float4 intrins = {focal_x, focal_y, float(W)/2.0, float(H)/2.0};
	glm::vec2 scale = scales[idx];
	glm::vec4 quat = rotations[idx];

	const float* transMat;
	bool ok;
	float3 normal;
	if (transMat_precomp != nullptr)
	{
		transMat = transMat_precomp + idx * 9;
	}
	else
	{
		ok = computeTransMat(p_world, quat, scale, viewmatrix, intrins, tan_fovx, tan_fovy, transMats + idx * 9, normal);
		if (!ok) return;
		transMat = transMats + idx * 9;
	}

	compute_weight(p_world, quat, scale, viewmatrix, weights[idx]);

	//  compute center and extent
	float2 center;
	float2 extent;
	ok = computeAABB(transMat, center, extent, W, H);
	if (!ok) return;

	// add the bounding of countour
#if TIGHTBBOX // no use in the paper, but it indeed help speeds.
	// the effective extent is now depended on the opacity of gaussian.
	float truncated_R = sqrtf(max(9.f + logf(opacities[idx]), 0.000001));
#else
	float truncated_R = 3.f;
#endif
	float radius = ceil(truncated_R * max(max(extent.x, extent.y), FilterSize));

	uint2 rect_min, rect_max;//高斯能影响到的栅格范围
	getRect(center, radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// compute colors
	if (colors_precomp == nullptr) {
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	depths[idx] = p_view.z;
	radii[idx] = (int)radius;
	points_xy_image[idx] = center;
	// store them in float4
	normal_opacity[idx] = {normal.x, normal.y, normal.z, opacities[idx]};
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);//影响tiles的个数
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching
// and rasterizing data.
template <uint32_t CHANNELS>//通道数
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)//每个tile由一个block处理，每个block中的每个线程处理每个像素
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	float focal_x, float focal_y,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float* __restrict__ transMats,
	const float* __restrict__ depths,
	const float4* __restrict__ normal_opacity,
	const float* __restrict__ weights,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	float* __restrict__ out_others,
	float* __restrict__ weight,//输出的权重张量
	float* __restrict__ transmittance,
	int* __restrict__ num_covered_pixels,
	bool record_transmittance)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();//当前线程块
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;//水平tiles个数
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };//当前线程对应的block负责处理的区域顶点像素
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };//当前线程负责处理的像素
	uint32_t pix_id = W * pix.y + pix.x;//像素的索引
	float2 pixf = { (float)pix.x + 0.5, (float)pix.y + 0.5};//像素的中心点坐标

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];//当前tile要处理的高斯起止索引
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);//整个线程块并行处理高斯的轮数
	int toDo = range.y - range.x;//需要处理的高斯总数

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float collected_weight[BLOCK_SIZE];//为高斯的权重分配共享内存
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_normal_opacity[BLOCK_SIZE];
	__shared__ float3 collected_Tu[BLOCK_SIZE];
	__shared__ float3 collected_Tv[BLOCK_SIZE];
	__shared__ float3 collected_Tw[BLOCK_SIZE];
	//线程块内部共享内存

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };


#if RENDER_AXUTILITY
	// render axutility ouput
	float D = { 0 };
	float N[3] = {0};
	float dist1 = {0};
	float dist2 = {0};
	float distortion = {0};
	float median_depth = {0};
	float median_weight = {0};
	float median_contributor = {-1};
	float Weight = {0};//记录当前像素的累计权重

#endif

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	//每次加载BLOCK_SIZE个高斯
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);//每个线程(即每个像素)是否都完成渲染
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();//线程加载数据的数据索引号
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];//加载的高斯点号
			collected_id[block.thread_rank()] = coll_id;//读入高斯点ID
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];//高斯的像素坐标
			collected_normal_opacity[block.thread_rank()] = normal_opacity[coll_id];//高斯的法线和透明度
			collected_weight[block.thread_rank()]=weights[coll_id];//加载指定高斯的权重
			collected_Tu[block.thread_rank()] = {transMats[9 * coll_id+0], transMats[9 * coll_id+1], transMats[9 * coll_id+2]};
			collected_Tv[block.thread_rank()] = {transMats[9 * coll_id+3], transMats[9 * coll_id+4], transMats[9 * coll_id+5]};
			collected_Tw[block.thread_rank()] = {transMats[9 * coll_id+6], transMats[9 * coll_id+7], transMats[9 * coll_id+8]};
		}
		block.sync();//线程块内部数据统一，同步所有线程
		//完成数据块中的数据加载，下面进行分线程的渲染

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;
			float gaussian_ray_weight=collected_weight[j];

			// Fisrt compute two homogeneous planes, See Eq. (8)
			float3 Tu = collected_Tu[j];
			float3 Tv = collected_Tv[j];
			float3 Tw = collected_Tw[j];//picf:像素中心点的坐标
			float3 k = {-Tu.x + pixf.x * Tw.x, -Tu.y + pixf.x * Tw.y, -Tu.z + pixf.x * Tw.z};
			float3 l = {-Tv.x + pixf.y * Tw.x, -Tv.y + pixf.y * Tw.y, -Tv.z + pixf.y * Tw.z};
			// cross product of two planes is a line (i.e., homogeneous point), See Eq. (10)
			float3 p = crossProduct(k, l);
#if BACKFACE_CULL
			// May hanle this by replacing a low pass filter,
			// but this case is extremely rare.
			if (p.z == 0.0) continue; // there is not intersection
#endif
			// 3d homogeneous point to 2d point on the splat
			float2 s = {p.x / p.z, p.y / p.z};
			// 3d distance. Compute Mahalanobis distance in the canonical splat' space
			float rho3d = (s.x * s.x + s.y * s.y);//像素点距离高斯中心的距离

			// Add low pass filter according to Botsch et al. [2005],
			// see Eq. (11) from 2DGS paper.
			float2 xy = collected_xy[j];
			float2 d = {xy.x - pixf.x, xy.y - pixf.y};
			// 2d screen distance
			float rho2d = FilterInvSquare * (d.x * d.x + d.y * d.y);
			float rho = min(rho3d, rho2d);//高斯局部坐标系下的3d距离+屏幕空间的2d距离联合判断

			float depth = (rho3d <= rho2d) ? (s.x * Tw.x + s.y * Tw.y) + Tw.z : Tw.z; // splat depth
			if ((s.x * Tw.x + s.y * Tw.y) + Tw.z < NEAR_PLANE) continue;//过滤掉太近的图元
			float4 nor_o = collected_normal_opacity[j];
			float normal[3] = {nor_o.x, nor_o.y, nor_o.z};//提取法线
			float power = -0.5f * rho;
			// power = -0.5f * 100.f * max(rho - 1, 0.0f);
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix).
			float alpha = min(0.99f, nor_o.w * exp(power));//高斯的不透明度
			if (record_transmittance){ //保存不透明度的信息
				atomicAdd(&transmittance[collected_id[j]], T * alpha);
				atomicAdd(&num_covered_pixels[collected_id[j]], 1);
			}
			if (alpha < 1.0f / 255.0f)
				continue;//高斯过于透明
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;//剩余的透明度过低，则完成渲染
				continue;
			}


#if RENDER_AXUTILITY
			// Render depth distortion map
			// Efficient implementation of distortion loss, see 2DGS' paper appendix.
			//像素点每渲染一个高斯透射率T就会被赋值T*(1-alpha)
			float A = 1-T;//T是透射率
			float mapped_depth = (FAR_PLANE * depth - FAR_PLANE * NEAR_PLANE) / ((FAR_PLANE - NEAR_PLANE) * depth);
			float error = mapped_depth * mapped_depth * A + dist2 - 2 * mapped_depth * dist1;
			distortion += error * alpha * T;

			if (T > 0.5) { //具有代表性的中值深度
				median_depth = depth;
				median_weight = alpha * T;
				median_contributor = contributor;
			}
			// Render normal map
			for (int ch=0; ch<3; ch++) N[ch] += normal[ch] * alpha * T;
			//根据每个高斯的权重逐渐叠加像素的法线

			// Render depth map
			D += depth * alpha * T;
			// Efficient implementation of distortion loss, see 2DGS' paper appendix.
			dist1 += mapped_depth * alpha * T;
			dist2 += mapped_depth * mapped_depth * alpha * T;

			Weight += gaussian_ray_weight * alpha * T;//计算累计权重
#endif

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;
			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;//像素最终的透光率
		n_contrib[pix_id] = last_contributor;//贡献该像素的高斯数目
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
			//剩余透光部分填充背景色

#if RENDER_AXUTILITY
		weight[pix_id]=Weight;//将累积的权重放入对应的像素中
		n_contrib[pix_id + H * W] = median_contributor;
		final_T[pix_id + H * W] = dist1;
		final_T[pix_id + 2 * H * W] = dist2;
		out_others[pix_id + DEPTH_OFFSET * H * W] = D;
		out_others[pix_id + ALPHA_OFFSET * H * W] = 1 - T;
		for (int ch=0; ch<3; ch++) out_others[pix_id + (NORMAL_OFFSET+ch) * H * W] = N[ch];
		out_others[pix_id + MIDDEPTH_OFFSET * H * W] = median_depth;
		out_others[pix_id + DISTORTION_OFFSET * H * W] = distortion;
		out_others[pix_id + MEDIAN_WEIGHT_OFFSET * H * W] = median_weight;
#endif
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	float focal_x, float focal_y,
	const float2* means2D,
	const float* colors,
	const float* transMats,
	const float* depths,
	const float4* normal_opacity,
	const float* weights,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	float* out_others,
	float* weight,
	float* transmittance,
	int* num_covered_pixels,
	bool record_transmittance)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		focal_x, focal_y,
		means2D,
		colors,
		transMats,
		depths,
		normal_opacity,
		weights,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		out_others,
		weight,
		transmittance,
		num_covered_pixels,
		record_transmittance);
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec2* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* transMat_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, const int H,
	const float focal_x, const float focal_y,
	const float tan_fovx, const float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* transMats,
	float* rgb,
	float4* normal_opacity,
	float* weights,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		transMat_precomp,
		colors_precomp,
		viewmatrix,
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		transMats,
		rgb,
		normal_opacity,
		weights,
		grid,
		tiles_touched,
		prefiltered
		);
}

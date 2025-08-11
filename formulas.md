# The formulas used in 2DGS rendering

This may not be displayed correctly on GitHub. Read it on other forms of visualization.

#### for every pixel $(x,y)$, every Gaussian:
$Tu=\begin{bmatrix}Tu_1\\Tu_2\\Tu_3\end{bmatrix}$ $Tv=\begin{bmatrix}Tv_1\\Tv_2\\Tv_3\end{bmatrix}$ $Tw=\begin{bmatrix}Tw_1\\Tw_2\\Tw_3\end{bmatrix}$
<br>

###### compute two homogeneous planes:
$k=\begin{bmatrix}Tu&Tw\end{bmatrix}\begin{bmatrix}-1\\x\end{bmatrix}$

$l=\begin{bmatrix}Tv&Tw\end{bmatrix}\begin{bmatrix}-1\\y\end{bmatrix}$

###### cross product of two planes is a line:
$p=k\times l=\begin{bmatrix}p_x\\p_y\\p_z\end{bmatrix}$

###### mapping the 2D pixels to 2D points under the Splat Coordinates:
$s=\begin{bmatrix}s_x\\s_y\end{bmatrix}=\begin{bmatrix}\dfrac{p_x}{p_z}\\ \\\dfrac{p_y}{p_z}\end{bmatrix}$
<br>

###### Mahalanobis distance between the 'Pixel' and the Gaussian Center in the canonical splat' space:
$\rho_{3d}={s_x}^2+{s_y}^2$

###### The distance between the 'Pixel' and the Gaussian Center on the screen:
$\rho_{2d}=(x_{Gaussian}-x)^2+(y_{Gaussian}-y)^2$

###### The final $\rho$ is the minimum of $\rho_{3d}$ and $\rho_{2d}$
<br>

$\alpha=\alpha_0\cdot \exp(-\dfrac{1}{2}\rho)$
accumulated opacity of a pixel: $A=1-\displaystyle \prod_{i=1}^{i=n-1}(1-\alpha_i)$

###### Depth Distortion:
distance to the far plane: $d_f$
distance to the near plane: $d_n$
$depth$ of a Gaussian is mapped to: $mapped\_depth=\dfrac{d_f\cdot depth-d_f\cdot d_n}{(d_f-d_n)\cdot depth}$(Normalization)
$distance_1=\displaystyle \sum_{i=1}^{i=n-1}(mapped\_depth_i\cdot \alpha_i\cdot \displaystyle\prod_{k=1}^{k=i-1}(1-\alpha_k))$
$distance_2=\displaystyle \sum_{i=1}^{i=n-1}(mapped\_depth_i^2\cdot \alpha_i\cdot \displaystyle\prod_{k=1}^{k=i-1}(1-\alpha_k))$
The distortion error of the $i_{th}$ Gaussian: $error_i=A\cdot mapped\_depth_i^2-2\cdot mapped\_depth_i\cdot distance_1+distance_2$
Total distortion error of the pixel: $error=\displaystyle \sum_{i=1}^{i=n}(error_i\cdot \alpha_i\cdot \displaystyle\prod_{k=1}^{k=i-1}(1-\alpha_k))$

###### Median Depth:
When the accumulated opacity reaches 0.5, store the Gaussian depth as the median depth
$median\_depth=depth_i$
$median\_weight=\alpha_i\cdot \displaystyle\prod_{k=1}^{k=i-1}(1-\alpha_k)$


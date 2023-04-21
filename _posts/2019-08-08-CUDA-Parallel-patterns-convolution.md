---
layout:     post
title:      CUDA-Parallel patterns convolution
subtitle:   Programming Massively Parallel Processors
date:       2019-08-08
author:     BY    J
header-img: img/post-bg-mma-4.jpg
catalog: true
tags:
    - CUDA
---

本章主要介绍了卷积作为一种重要的并行计算模式。首先展示了基本的并行卷积算法，其执行速度受到了DRAM带宽的限制。然后提出了tiling kernel。并介绍了一种数据缓存的简化版代码。最后介绍了2D卷积核。
在高性能计算中，卷积通常称为stencil computation。因为每一个输出元素均可以独立计算，且输入数据共享了输出元素，这些特性使得卷积成为复杂tiling 方法与输入数据分布策略的重要应用。
缺失的边界值(padding)通常称为"ghost cells" or "halo cells"。这些cell值对tiling的高效性。

## 1D Parallel Convolutional-A Basic Algorithm
```
__global__ void convolution_1D_basic_kernel(float *N, float *M, float *P, int Mask_Width, int Width){
//kenel body
}
```
假设Mask_width是奇数，卷积对称。例如：Mask_Width = 2*n + 1,则P[i]的输出值将利用N[i-n], N[i-n+1],...，N[i],N[i+1],N[i+n-1],N[i+n]。
```
float Pvalue = 0;
int N_start_point = i - (Mask_Width/2);
for(int j =0;j < Mask_Width; j++)
{
if(N_start_point + j >=0 && N_start_point + j < Width){
Pvalue += N[N_start_point + j] *M[j];
}
}
P[i] = Pvalue;
```
上述省略了ghost cell与对应N值的乘积
核函数：
```
__global__ void convolution_1D_basic_kernel(float *N, float *M, float *P, int Mask_Width, int Width){

int i = blockIdx.x * blockDim.x + threadIdx.x;
float Pvalue = 0;
int N_start_point = i - (Mask_Width / 2);
for(int j =0;j < Mask_Width;j++){
if(N_start_point + j >= 0 && N_start_point + j < Width){
Pvalue += N[N_start_point + j]*M[j];
}
}
P[i] = Pvalue;
}
```
1D convolution kernel with boundary condition handling
分析：上述会有control flow divergence. 其代价视输入数组的宽度与mask的Mask_Width的不同而变化。
比较严重的问题是内存带宽。这里浮点数学计算对全局内存的访问比值才为1.
## Constant Memory and Caching
滤波器数组M的三个有趣属性。1是M一般很小。2是M的元素在程序执行时不变。3是所有的线程都以相同顺序访问M的元素。这些特性使得Mask array非常适合constant memory与caching。
constant memory在核函数执行过程中不会改变，比较小，当前只有64KB.
```
#define Max_MASK_WIDTH 10
__constant__ float M[Max_MASK_WIDTH];
```
transfer code
```
cudaMemcpyToSymbol(M, M_h, Mask_Width * sizeof(float));
```
consant memory变量作为全局变量传给核函数。
为了理解应用constant memory的益处，我们需要更多的理解现代处理器内存与cache hierarchies。
在第5章中，我们知道长延时与DRAM的限制性的带宽是现在处理器的主要瓶颈。为了消除内存瓶颈，现代处理器通常应用片上高速缓存，来减少需要从主内存(DRAM)上访问的变量数量。如下图：
![7-1](https://res.cloudinary.com/dsn0i1fsm/image/upload/v1563888676/blog/CUDA/7part/9_mva9pq.jpg)
处理器将自动在高速缓存中保留最常用的变量并记住它们的DRAM地址。之后当变量用到时，高速缓存中将复制变量。
L1,L2层级高速缓存。L1速度上应该是最快的。
其中一个设计时需要考虑的就是cache coherence。一级缓存修改数据时，其他层级的缓存时不容易发现的。所以需要cache coherence mechanism。大部分CPU均支持。GPU为了增加处理器吞吐量，一般没有该机制。
Constant memory因为在执行过程中不允许修改变量，所以也不需要关注cache coherence。
## Tiled 1D Convolution with Halo Cells
这里每一个block处理的输出元素的集合称为output tile。下图展示了用4个block（每个block4个线程）处理1D卷积的例子。在实践中，每个block至少要有32个线程。我们假设M个元素存储在constant memory中。
![7-2](https://res.cloudinary.com/dsn0i1fsm/image/upload/v1563888676/blog/CUDA/7part/10_g9qohj.jpg)
为了减少全局内存的总体访问数量，我们将讨论两种输入数据的tiling策略。第一种是将所有需要计算一个线程block中的输出元素的输入数据加载进共享内存中。需要加载的输入元素大小取决于mask的大小。这里我们假设size = 2 * n + 1。上图中，Mask_Width = 5, n = 2。
Block 0中的线程计算P[0]到P[3]的输出元素。输入元素则需要N[0]到N[5]。另外，还需要N[0]左侧的两个ghost cell元素。其默认值为0。Tile 3右侧也是同样的情形。这里，我们将类似Tile 0 与Tile 3这种成为**边界tiles**。
Block 1中的线程计算输出元素P[4]到P[7]。他们需要输入元素N[2]到N[9]。元素N[2]到N[3]属于两个tiles并且两次被加载到共享内存中。一次是Block[0],一次是Block[1]。每个block中共享内存的内容仅对自己的线程可见，所以需要被加载进各自的共享内存。像这种被多个tiles涉及并加载多次的元素称为**halo cells或者skirt cells**。输入tile中仅被一个block应用的部分称为**internal cells**。
下面将展现将输入tile加载进共享内存的核函数代码。
首先定义共享内存数组N_ds以保存每个block的tile。确保N_ds足够大。
```
__shared__ float N_ds[TILE_SIZE + MAX_MASK_WIDTH - 1];
```
然后，我们加载左边的halo cells。
```
int halo_index_left = (blockIdx.x - 1) * blockDim.x + threadIdx.x; // Line 1
if(threadIdx.x >= blockDim.x - n){
N_ds[threadIdx.x - (blockDim.x - n)] = 
(halo_index_left < 0)? 0:N[halo_index_left]; // Line 2
}
```
上述代码Line 1将线程映射到之前tile的元素索引。然后挑选最后的n个线程加载需要的左边halo元素。例如blockDim.x = 4，则只有2和3线程用到了。
Line 2则用于检查halo cells是否是ghost cells。
下一步则是加载输入tile的中间cells。
```
N_ds[n + threadIdx.x] = N[blockIdx.x * blockDim.x + threadIdx.x];
```
N_ds的头n个元素包含了left halo cells，则中间元素需要加载进N_ds的下一部分。
然后是加载右边的halo元素,下面我们加载下一个tile的头n个元素。
```
int halo_index_right = (blockIdx.x + 1)*blockDim.x + threadIdx.x;
if(threadIdx.x < n){
N_ds[n + blockDim.x + threadIdx.x] = 
(halo_index_right > width ) ? 0: N[halo_index_right];
}
```
N_ds现在存储有所有的输入tile。每一个线程将利用N_ds的不同部分计算对应的P值。
```
float Pvalue = 0;
for(int j = 0; j < Mask_Width; j++){
Pvalue += N_ds[threadIdx.x + j] * M[j];
}
P[i] = Pvalue;
```
这里需要加__syncthreads()以确保相同block的所有线程已经加载完毕所需元素。
![7-3](https://res.cloudinary.com/dsn0i1fsm/image/upload/v1563888676/blog/CUDA/7part/11_ayrsrc.jpg)
tiled 1D卷积版本比正常卷积核函数更加复杂，这里引进了额外的算法复杂度以减少DRAM访问的次数，目的则是提高访问计算比。
原文中关于优化访问量的分析，请读原文。
## A Simpler Tiled 1D Convolution-General Caching
L1对于每个流处理器是私有的，L2对于所有的流处理器是共享的。该特性可以使我们利用halo cells存在L2缓存中这一特点。
一个block的halo cell值可能是邻近block的内部cell。Tile 1中的halo cell N[2]和N[3]则是Tile 0的内部cell。**Block 1需要利用这些值，但是由于Block 0的访问，它们已经存在于L2 缓存中。**因此我们可以不用将这些值加载进N_ds。下面将呈现更简单的1D 卷积，其仅加载每个tile的内部元素进入共享内存。
在更简单的tiled kernel中，共享内存N_ds数组只需要加载tile的内部元素。因此大小为声明为TILE_SIZE，而不是TILE_SIZE + MASK_WIDTH -1.
```
__shared__ float N_ds[TILE_SIZE];
N_ds[threadIdx.x] = N[blockIdx.x * blockDim.x + threadIdx.x];
```
This_tile_start_point与Next_tile_start_point控制了边界，边界条件内，则用N_ds,否则从N中取值。
```
__global__ void convolution_1D_tiled_caching_kernel(float* N, float* P,
int Maks_Width, int Width)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
__shared__ float N_ds[TILE_SIZE];

N_ds[threadIdx.x] = N[i];
__syncthreads();
int This_tile_start_point = blockIdx.x * blockDim.x;
int Next_tile_start_point = (blockIdx.x + 1) * blockDim.x;
int N_start_point = i - (Mask_Width / 2);
for(int j = 0; j < Mask_Width; j++)
{
 int N_index = N_start_point + j;
 if(N_index > 0 && N_index < Width)
 {
 if((N_index > This_tile_start_point) && (N_index < Next_tile_start_point))
 {
 Pvalue += N_ds[threadIdx.x + j - (Mask_width / 2)] * M[j];
 }else{
 Pvalue += N[N_index] * M[j];
 }
 }
}
P[i] = Pvalue;
}
```
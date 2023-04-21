---
layout:     post
title:      CUDA-Memory and data locality
subtitle:   Programming Massively Parallel Processors
date:       2019-07-01
author:     BY    J
header-img: img/post-bg-mma-4.jpg
catalog: true
tags:
    - CUDA
---

本章总结了共享内存的用法以及注册器可以影响每个流多处理器能容纳的线程BLOCK的数量---memory bound.
迄今为止我们了解的知识只是挖掘了潜在硬件性能的很小一部分.程序的糟糕表现主要是由于全局内存的长访问延迟和有限的获取带宽,通常是由Dynamic Random Access Memory.(DRAM).
内存高效访问的重要性
compute-to-global-memory-access（CGMA） is defined as the number of floating-point calculation performed for each access to the global memory within a region of progam.
浮点计算对应全局内存获取的比值
computer-to-global-memory-access对CUDA核的表现有重要影响.程序的执行速度被内存获取吞吐量限制了.为了获得更高的表现,我们需要通过减少全局内存的获取来增加ratio.
```
__global__ void MatrixMulkernel(float* M, float* N, float* P, int Width){
int Col = blockIdx.x * blockDim.x + threadIdx.x;
int Row = blockIdx.y * blockDim.y + threadIdx.y;

if( (Row < Width) && (COl < Width))
{
float Pvalue = 0;
for(int k = 0; k < Width; ++k){
Pvalue += M[Row * Width + k] *N[K* Width + Col];
}
P[Row * Width + Col] = Pvalue;
}
}
```
对于矩阵运算,矩阵M和N对元素的获取是两次全局内存访问对应一次浮点加法与乘法.比率为1.此比率将导致现代GPU少于峰值执行速度利用率的2%.我们需要增加设备的计算吞吐量
Registers and shared memory(寄存器与共享内存)是on-chip memories。存储在这两者上的变量可以高速并行的方式获取。每一个线程拥有自己的寄存器。核函数通常用寄存器存储高频访问的线程私有变量。共享内存被分配给线程块。一个线程块中的所有内存都可以获取到共享内存变量。共享内存是线程协作的高效方式。
![display](https://res.cloudinary.com/dsn0i1fsm/image/upload/v1561959180/blog/CUDA/memory_ajar5z.jpg)

内存类型的了解
![display](https://res.cloudinary.com/dsn0i1fsm/image/upload/v1561959731/blog/CUDA/v1_o4wovv.jpg)
1. Globay memory在处理器芯片外，由DRAM技术实现，意味着长时间访问延迟和低访问带宽。Register File通常在寄存器上，意味着很短的访问延迟和极高的访问带宽。在通常设备上，访问带宽的性能是Global memory的至少两倍数量级。当变量存储在Register上时，它的访问不再消耗global memory。减少的带宽消耗将通过增加的CGMA比值所反映出来。
2. 操作更少，因为不用再执行加载。
3. 从Register上访问变量消耗的能量也比global至少少一个数量级。但是每个线程可用的Register数量相当受限。
共享内存与Register同属于片上内存。但是共享内存需要执行内存加载操作，所以性能上Register > shared memory > global memory
![display](https://res.cloudinary.com/dsn0i1fsm/image/upload/v1561959731/blog/CUDA/v2_c1crjs.jpg)
一个重要不同就是block块中的所有线程可以访问共享内存中的变量，但是Register中为线程的私有变量。
![display](https://res.cloudinary.com/dsn0i1fsm/image/upload/v1561963012/blog/CUDA/memory_type_dugzde.png)

1.标量(非数组或矩阵变量)在核函数或者device function中被分类到Register中。变量blurRow, blurCol,curRow,curCol,pixels,pixVal均归到此类。需要注意的是Register的存储容量，不要超出限制。大量的Register会严重影响分配给每个SM的活跃线程数量。
2.数组变量分配在global memory中。在核函数与设备函数中很少用。

"__constant__"代表了常量内存。常量内存必须在函数体外。其生命周期就是程序的生命周期。其存储在全局内存中，但是为了高效访问是以缓存形式存在的。常量内存大小限制在65536个字节。所以输入需要分开喂入程序，如第7章卷积部分所展示的那样。
变量声明前只有"__device__"的是一个全局变量并且将会放置在全局内存中。对全局内存的访问很慢。其生命周期也是伴随着整个应用进程。因此其可以用来在不同线程块之间进行协作。不同线程之间需要同步来确保信息的一致性。
在CUDA中，指针用来指向全局内存中的数据对象。在核函数和设备函数中指针的用法有两种方式，一是对象都有host function创建，则由cudaMalloc()初始化，并作为参数传递给核函数。二是在全局内存中声明的变量地址被赋值给指针变量。读者应该参考CUDA 编程文档来参考其他的内存类型。

##  4.4 Tiling for Reduced Memory Traffic
全局内存大但是慢，而共享内存小但是快。通用的策略是将数据分成叫做tiles的子集，这样每一个tile可以适应共享内存。一个重要的标准是在这些tiles上的核函数计算可以相互独立执行。需要说明的是给定任意核函数，并不是所有数据结构可以分成tiles.
下面将通过4.5中矩阵乘法的例子来解释**tiling**的概念.

![display](https://res.cloudinary.com/dsn0i1fsm/image/upload/v1561988864/blog/CUDA/4-8_wwahaj.jpg)
此示例假设我们用2 * 2 的block来计算P矩阵. 下图展示了由block(0, 0)的4个线程执行的计算过程.这4个线程分别计算P(0,0),P(0,1),P(1,0)和P(1,1).由block(0, 0)的线程(0,0)和线程(0, 1)访问的M与N中的元素由黑色箭头进行标示.例如thread(0, 0)读取M(0,0),N(0,0),M(0,1),N(1,0),M(0,2),N(2,0),M(0,3),N(3,0).
![display](https://res.cloudinary.com/dsn0i1fsm/image/upload/v1561989324/blog/CUDA/4-9_nsj6ga.jpg)
图4.10展示了block(0,0)中所有线程访问的全局内存.表格中竖直方向表示了线程,水平方向展示了随时间增加对矩阵中元素的访问顺序.每一个线程在执行过程中分别访问矩阵M与N的4个元素.但是在4个线程中,线程访问之间有很多的重叠部分.例如,thread(0,0)与thread(0,1)都访问了M(0,0)以及M中行位置是0的元素.
![display](https://res.cloudinary.com/dsn0i1fsm/image/upload/v1561989348/blog/CUDA/4-10_jglayl.jpg)



如果thread(0, 0) 与thread(0, 1)可以协同工作,那么M中的元素就可以只加载一次.全局内存总的访问量可以降低一半.

In the context of parallel computing, tiling is a program transformation technique that localizes the memory locations accessed among threads and the timing of their accesses. It divides the long access sequences of each thread into phases and uses barrier synchronization to keep the timing of accesses to each section at close intervals.

现在将介绍tiled矩阵乘法算法.基本的想法是在他们单独计算内积计算时,线程协作加载M与N元素的子集到共享内存.将M与N划分成较小的tiles,可以使它们加载进共享内存.最简单的形式就是tile的维度等于block的维度.

![display](https://res.cloudinary.com/dsn0i1fsm/image/upload/v1561992258/blog/CUDA/4-13_az1zpl.jpg)



![display](https://res.cloudinary.com/dsn0i1fsm/image/upload/v1561992266/blog/CUDA/4-14_hwxlmn.jpg)

上图中Phase1中N(0,1)应该对应于Nds(0, 1).

对于BLOCK(0,0)计算分成了两部分,观察图,可以知道取中间值Pvalue,通过累加实现了计算.

如果矩阵的宽度为Width, tiles的大小为TILE_WIDTH,则点积的计算将在WIDTH/TILE_WIDTH阶段内完成.

 说明: Mds  和Nds重复利用来保存输入值. This is due to the fact that each phase focuses on a small subset of the input matrix elements. Such focused access behaviour is called locality.

**Locality is as important for achieving high-performance in multi-core CPUs as in many-thread GPUs.**

## 4.5 A Tiled Matrix Multiplication Kernel

```
__global__ void MatrixMulKernel(float *d_M, float* d_N, float* d_P, int Width){
__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

int bx = blockIdx.x; int by = blockIdx.y;
int tx = threadIdx.x; int ty = threadIdx.y;
// Identify the row and column of the d_p element to work on
int Row = by * TILE_WIDTH + ty;
int Col = bx * TILE_WIDTH + tx;
float Pvalue = 0;
for(int ph = 0; ph < Width / TILE_WIDTH; ph++){
	Mds[ty][tx] = d_M[Row * WIDTH +ph* TILE_WIDTH + tx];
	Nds[ty][tx] = d_N[(ty + ph * TILE_WIDTH) * WIDTH + Col];
	__syncthreads();
}
for(int k = 0; k < TILE_WIDTH; k++)
{
pvalue += Mds[ty][k] * Nds[k][tx];
}
__syncthreads();
d_P[Row * Width + Col] = Pvalue;
}
```

![Display](https://res.cloudinary.com/dsn0i1fsm/image/upload/v1561996239/blog/CUDA/4-15_bjjdta.jpg)

可以结合上图对函数进行理解,主要是计算坐标的理解.

第一个__synchreads()确保了所有线程完成了tiles的加载.第二个则确保所有线程用完共享内存的元素.

Strip-mining技术.取长循环为分阶段实现.如果用16*16的TILE_WIDTH,则可以减少全局内存的访问16倍.即增加CGMA由1到16.

上述有两个假设,首先矩阵的宽度是BLOCK宽度的倍数.二是矩阵必须是方形矩阵.下面一节将打破这两个假设.

## 4.6 Boundary Checks

我们现在扩展这个矩阵乘法算法到可以处理任意的矩阵宽度.这次扩展可以使核能处理任意宽度不是tile宽度的倍数的情况.下面依然是方形矩阵.我们需要检查输入矩阵的有效情况.边界检查:

```
Row < WIDTH && (ph * TILE_WIDTH + tx) < WIDTH)
(ph * TILE_WIDTH + ty) < WIDTH && Col < WIDTH
```
代码:
```
for(int ph = 0; ph < ceil(Width / (float)TILE_WIDTH; ++ph)
{
 if((ROW < WIDTH)&&(ph * TILE_WIDTH + tx) < Width)
     Mds[ty][tx] = M[Row * Width + ph * TILE_WIDTH + tx];
 if((ph * TILE_WIDTH + ty) < Width && Col < Width)
     Nds[ty][tx] = N[(ph*TILE_WIDTH + ty)*Width + Col];
 __syncthreads();
 for(int k = 0; k < TILE_WIDTH; ++k){
  Pvalue += Mds[ty][k] * Nds[k][tx];
 }
 __syncthreads();
}
if((Row < Width) && (Col < Width) P[Row * Width + Col] = Pvalue;)
```
下面距离通用的矩阵乘法核还差一步,对于J* K的M矩阵与K * L的N矩阵相乘.
```
M: Col < J && (ph * TILE_WIDTH + tx) < K
N: (ph * TILE_WIDTH + ty) < K && Col < L
```
## 4.7 Memory as a Limiting Factor to Parallelism
通常每个线程要求的资源越多,每个SM中存在的线程则越少.
cudaGetDeviceProperties(&dev_prop)
dev_prop.regsPerBlock 查看register的数量
对于共享内存也是,对于矩阵乘法来说,对于16 * 16的tile_size, 则每个block需要16 * 16 *4 = 1k字节的存储(说明 每个元素是浮点数, 大小为4个字节).加上Nds, 则需要2K.所以 一个16K的共享内存只允许 同时存在8个block.在这里,真正的限制是线程数量.如果每个SM有1536个线程,则只允许存在6个block.所以只能用6 * 2KB = 12KB的共享内存.
每一代的共享内存数量都不同,我们可能希望代码可以随意调整一个kernel中的共享内存大小. 
通过cudaGetDeviceProperties(&dev_prop), dev_prop.sharedMemPerBlock.

**extern  与size_t需要注意.以后需要回看**


















































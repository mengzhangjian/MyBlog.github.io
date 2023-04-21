---
layout:     post
title:      CUDA - 流(Stream)
subtitle:   CUDA学习
date:       2019-06-22
author:     BY    J
header-img: img/home-bg-o.jpg
catalog: true
tags:
    - CUDA	
---

## 本章目标
+ 了解如何分配页锁定（Page-Locked）的主机内存
+ 了解CUDA流的概念
+ 了解如何使用CUDA流来加速应用程序

malloc()与cudaHostAlloc()分配的内存之间存在着一个巨大的差异。C库函数malloc()将分配标准的，可分页的主机内存。而cudaHostAlloc()将分配页锁定的主机内存。页锁定内存也称为固定内存（Pinned Memory）或者不可分页内存，它有一个重要属性：操作系统将不会对这块内存分页并交换到磁盘上，从而确保了该内存始终驻留在物理内存中。
由于 GPU知道内存的物理地址，因为可以通过直接内存访问（Direct Memory Access）DMA技术在GPU和主机之间复制数据。
每当从可分页内存中执行复制操作时，复制速度将受限于PCIE的传输速度和系统前端总线速度相对较低的一方。
如果将所有malloc进行替换，则会更快的耗尽系统内存。
建议：仅对cudaMemcpy()调用中的源内存或者目标内存，才使用页锁定内存，并且在不需要它们时进行立即释放。

## 测试程序
测试目标： 主要测试cudaMemcpy()在可分配内存和页锁定内存上的性能。分配一个GPU缓冲区以及一个大小相等的主机缓冲区，然后在这两个缓冲区之间执行复制操作。并设置CUDA事件进行精确的时间统计。
```
float cuda_malloc_host_test(int size, bool up){

    int *a, *dev_a;
    float elapsedTime;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaHostAlloc( (void**)&a, size * sizeof(*a), cudaHostAllocDefault);
    cudaMalloc((void**)&dev_a, size * sizeof(*dev_a));

    cudaEventRecord(start, 0);
    for(int i = 0; i < 100; i++){

        if(up)
            cudaMemcpy(dev_a, a, size * sizeof(*a), cudaMemcpyHostToDevice);
        else
            cudaMemcpy(a, dev_a, size * sizeof(*dev_a), cudaMemcpyDeviceToHost);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaFreeHost(a);
    cudaFree(dev_a);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return elapsedTime;
}
```
以上可以看到，cudaHostAlloc()与malloc()分配的内存在使用方式上是相同的，不同的是最后一个参数cudaHostAllocDefault.我们可以通过修改该参数来修改cudaHostAlloc()的行为，并分配不同形式的固定主机内存。如果要释放内存，必须使用cudaFreeHost()。

## CUDA流
CUDA流表示一个GPU操作队列，并且队列上的操作将以指定的顺序执行。我们可以在流中添加一些操作。将这些操作添加的顺序就是它们的执行顺序。你可以将每个流视为GPU上的一个任务，并且这些任务可以并行执行。
### 使用单个CUDA流
首先选择一个支持设备重叠功能的设备（Device Overlap）.支持设备重叠功能的GPU能够在执行一个CUDA核函数的同时,还能在设备与主机之间执行复制操作. 我们将使用多个流来实现这种计算与数据的传输的重叠,但首先来看如何创建和使用一个流.

此次采用cudaMemcpyAsync()在GPU与主机之间复制数据.cudaMemcpy()函数是以同步的方式进行的,当函数返回时,复制操作已经完成.**任何传递给cudaMemcpyAsync()的主机内存指针都必须已经通过cudaHostAlloc()分配好内存.也就是,你只能以异步的方式对页锁定内存进行复制操作.**

调用cudaStreamSynchronize()指定想要等待的流.

cudaStreamDestroy()销毁流.

## 使用多个CUDA流
在任何支持内存复制和核函数执行相互重叠的设备上,当使用多个流时,应用程序的性能都会整体提升.
建立两个流,stream0 和stream1两个流交替执行,实验测试性能确实提升.
```
  for(int i = 0; i < FULL_DATA_SIZE;i += N * 2){
        cudaMemcpyAsync(dev_a0, host_a + i, N* sizeof(int), cudaMemcpyHostToDevice, stream0);
        cudaMemcpyAsync(dev_b0, host_b + i, N* sizeof(int), cudaMemcpyHostToDevice, stream0);
        kernel<<<N/256, 256, 0, stream0>>>(dev_a0, dev_b0, dev_c0);
        cudaMemcpyAsync(host_c + i, dev_c0, N* sizeof(int), cudaMemcpyDeviceToHost, stream0);

        cudaMemcpyAsync(dev_a1, host_a + i + N, N* sizeof(int), cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(dev_b1, host_b + i +N, N* sizeof(int), cudaMemcpyHostToDevice, stream1);
        kernel<<<N/256, 256, 0, stream0>>>(dev_a1, dev_b1, dev_c1);
        cudaMemcpyAsync(host_c + i + N, dev_c1, N* sizeof(int), cudaMemcpyDeviceToHost, stream1);
}
```
## GPU的工作调度机制
需要确切理解GPU中流的执行机制,逻辑上来讲,不同流的运行是相互独立的,但是硬件上并没有流的概念,而是包含一个或多个引擎来执行内存的操作复制,以及执行核函数的引擎.**书中图10.4,使用多个GPU流时的程序执行时间线,由于第0个流中将C复制回主机需要等待核函数执行完成才能进行,因此第1个流中虽然将a和b复制到GPU 的操作是完全独立的,但全被阻塞了.**因此我们需要知道将操作放入流中的顺序将影响着CUDA驱动程序调度这些操作以及执行的方式.
## 高效地使用多个CUDA流
要解决上述问题,在将操作放入流的队列时应采用宽度优先的方式,而非深度优先的方式.也就是说,我们并不是一次性添加第0个流的所有操作,而是两个流的操作交替进行添加.
```
  for(int i = 0; i < FULL_DATA_SIZE;i += N * 2){
        cudaMemcpyAsync(dev_a0, host_a + i, N* sizeof(int), cudaMemcpyHostToDevice, stream0);
        cudaMemcpyAsync(dev_b0, host_b + i, N* sizeof(int), cudaMemcpyHostToDevice, stream0);
        cudaMemcpyAsync(dev_a1, host_a + i + N, N* sizeof(int), cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(dev_b1, host_b + i +N, N* sizeof(int), cudaMemcpyHostToDevice, stream1);
        kernel<<<N/256, 256, 0, stream0>>>(dev_a0, dev_b0, dev_c0);
        kernel<<<N/256, 256, 0, stream0>>>(dev_a1, dev_b1, dev_c1);      
        cudaMemcpyAsync(host_c + i, dev_c0, N* sizeof(int), cudaMemcpyDeviceToHost, stream0);
        cudaMemcpyAsync(host_c + i + N, dev_c1, N* sizeof(int), cudaMemcpyDeviceToHost, stream1);

    }
```
执行顺图可看书中图10.5,也可比较上述两段代码.
## 总结
本节介绍了在CUDA C程序中实现任务级的并行性.通过使用多个CUDA流,我们可以使GPU在执行核函数的同时,还能在主机和GPU之间执行复制操作.然而,采用这种方式时,需要注意里两个因素.首先,需要通过cudaHostAlloc()分配主机内存,因此接下来需要通过cudaMemcpyAsync()对内存复制操作进行排队,而异步复制操作需要在固定缓冲区执行.其次,我们需要知道添加操作到流中的顺序会对重叠情况产生影响.通常,应该采用宽度优先或者轮询的方式将工作分配到流.

































































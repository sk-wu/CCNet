#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>

#include "common.h"
#include "ca.h"


__global__ void ca_forward_kernel(const float *t, const float *f, float *weight, int num, int chn, int height, int width) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int sp = height * width;
  int len = height + width - 1;
  int z = blockIdx.z;

  if (x < width && y < height && z < height+width-1) {
    for (int batch = 0; batch < num; ++batch) {
      for (int plane = 0; plane < chn; ++plane) {
        float _t = t[(batch * chn + plane) * sp + y*width + x]; # 锁定（x,y）点的第plane个通道
        
        if (z < width) {
          int i = z;
          float _f = f[(batch * chn + plane) * sp + y*width + i]; # 锁定(x, y)点水平方向第i个点的第plane个通道
          weight[(batch * len + i) * sp + y*width + x] += _t*_f; # weight尺寸为（c,h+w-1, h, w）
        } else {
          int i = z - width;
          int j = i<y ? i : i+1;

          float _f = f[(batch * chn + plane) * sp + j*width + x]; # 锁定(x, y)点竖直方向第j个点的第plane个通道
          weight[(batch * len + width + i) * sp + y*width + x] += _t*_f;
        }
      }
    }
  }
}

__global__ void ca_backward_kernel_t(const float *dw, const float *t, const float *f, float *dt,
                                int num, int chn, int height, int width) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int sp = height * width;
  int len = height + width - 1;
  int plane = blockIdx.z;

  if (x < width && y < height && plane < chn) {
    for (int batch = 0; batch < num; ++batch) {
        
        for (int i = 0; i < width; ++i) {
          float _dw = dw[(batch * len + i) * sp + y*width + x]; # 权重矩阵(x,y)点的第i个通道的值（表示(x,y)点和第i个点的关联），(x,y)点共有(h+w-1)个通道,这里先算w个
          float _f = f[(batch * chn + plane) * sp + y*width + i]; # 点(x,y)关联的第i个点在通道z上的值,i为水平方向的点
          dt[(batch * chn + plane) * sp + y*width + x] += _dw * _f; # 将关联还原到每个通道上
        } 
        for (int i = 0; i < height; ++i)  {
          if (i == y) continue;
          int j = i<y ? i : i-1;

          float _dw = dw[(batch * len + width + j) * sp + y*width + x]; # 权重矩阵(x,y)点的第(width+j)个通道上的值（表示(x,y)点和第j个点的关联），(x,y)点共有(h+w-1)个通道，这里算h个
          float _f = f[(batch * chn + plane) * sp + i*width + x]; # 点(x,y)关联的第i个点在通道z上的值,i为垂直方向的点
          dt[(batch * chn + plane) * sp + y*width + x] += _dw * _f; # 将关联还原到每个通道上
        }
    }

  }
}

__global__ void ca_backward_kernel_f(const float *dw, const float *t, const float *f, float *df, 
                                int num, int chn, int height, int width) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int sp = height * width;
  int len = height + width - 1;
  int plane = blockIdx.z;

  if (x < width && y < height && plane < chn) {
    
    for (int batch = 0; batch < num; ++batch) {
      
      for (int i = 0; i < width; ++i) {
        float _dw = dw[(batch * len + x) * sp + y*width + i]; # 找出某个位置x通道上水平和垂直方向的（h+w-1）个点
        float _t = t[(batch * chn + plane) * sp + y*width + i]; # 找出第plane个通道水平和垂直方向的（h+w-1）个点，将梯度反向传递回去
        df[(batch * chn + plane) * sp + y*width + x] += _dw * _t;
      }
      for (int i = 0; i < height; ++i) {
        if (i == y) continue;
        int j = i>y ? y : y-1;

        float _dw = dw[(batch * len + width + j) * sp + i*width + x];
        float _t = t[(batch * chn + plane) * sp + i*width + x];
        df[(batch * chn + plane) * sp + y*width + x] += _dw * _t;
      }
    }

  }
}


__global__ void ca_map_forward_kernel(const float *weight, const float *g, float *out, int num, int chn, int height, int width) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int sp = height * width;
  int len = height + width - 1;
  int plane = blockIdx.z;

  if (x < width && y < height && plane < chn) {
    for (int batch = 0; batch < num; ++batch) {

      for (int i = 0; i < width; ++i) {
        float _g = g[(batch * chn + plane) * sp + y*width + i];
        float _w = weight[(batch * len + i) * sp + y*width + x];
        out[(batch * chn + plane) * sp + y*width + x] += _g * _w;
      }
      for (int i = 0; i < height; ++i) {
        if (i == y) continue;

        int j = i<y ? i : i-1;

        float _g = g[(batch * chn + plane) * sp + i*width + x];
        float _w = weight[(batch * len + width + j) * sp + y*width + x];
        out[(batch * chn + plane) * sp + y*width + x] += _g * _w;
      }
    }
  }

}

__global__ void ca_map_backward_kernel_w(const float *dout, const float *weight, const float *g, float *dw,
                                int num, int chn, int height, int width) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int sp = height * width;
  int len = height + width - 1;
  int z = blockIdx.z;

  if (x < width && y < height && z < height+width-1) {

    for (int batch = 0; batch < num; ++batch) {
      for (int plane = 0; plane < chn; ++plane) {
        float _dout = dout[(batch * chn + plane) * sp + y*width + x];

        if (z < width) {
          int i = z;
          float _g = g[(batch * chn + plane) * sp + y*width + i];
          dw[(batch * len + i) * sp + y*width + x] += _dout * _g;
        } else {
          int i = z - width;
          int j = i<y ? i : i+1;

          float _g = g[(batch * chn + plane) * sp + j*width + x];
          dw[(batch * len + width + i) * sp + y*width + x] += _dout * _g;
        }
      }
    }
  }
}

__global__ void ca_map_backward_kernel_g(const float *dout, const float *weight, const float *g, float *dg, 
                                int num, int chn, int height, int width) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int sp = height * width;
  int len = height + width - 1;
  int plane = blockIdx.z;

  if (x < width && y < height && plane < chn) {

    for (int batch = 0; batch < num; ++batch) {
      for (int i = 0; i < width; ++i) {
        float _dout = dout[(batch * chn + plane) * sp + y*width + i];
        float _w = weight[(batch * len + x) * sp + y*width + i];
        dg[(batch * chn + plane) * sp + y*width + x] += _dout * _w;
      }

      for (int i = 0; i < height; ++i) {
        if (i == y) continue;
        int j = i>y ? y : y-1;

        float _dout = dout[(batch * chn + plane) * sp + i*width + x];
        float _w = weight[(batch * len + width + j) * sp + i*width + x];
        dg[(batch * chn + plane) * sp + y*width + x] += _dout * _w;
      }
    }
  }
}

/*
 * Implementations
 */
extern "C" int _ca_forward_cuda(int N, int C, int H, int W, const float *t, 
                                const float *f, float *weight, cudaStream_t stream) {
  // Run kernel
  dim3 threads(32, 32);
  int d1 = (W+threads.x-1)/threads.x;
  int d2 = (H+threads.y-1)/threads.y;
  int d3 = H+W;
  dim3 blocks(d1, d2, d3);
  ca_forward_kernel<<<blocks, threads, 0, stream>>>(t, f, weight, N, C, H, W);

  // Check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    return 0;
  else
    return 1;
}


extern "C" int _ca_backward_cuda(int N, int C, int H, int W, const float *dw, const float *t, const float *f, float *dt, float *df, cudaStream_t stream) {
  // Run kernel
  dim3 threads(32, 32);
  int d1 = (W+threads.x-1)/threads.x;
  int d2 = (H+threads.y-1)/threads.y;
  int d3 = C;
  dim3 blocks(d1, d2, d3);
  // printf("%f\n", dw[0]);
  ca_backward_kernel_t<<<blocks, threads, 0, stream>>>(dw, t, f, dt, N, C, H, W);
  ca_backward_kernel_f<<<blocks, threads, 0, stream>>>(dw, t, f, df, N, C, H, W);

  // Check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    return 0;
  else
    return 1;
}


extern "C" int _ca_map_forward_cuda(int N, int C, int H, int W, const float *weight, const float *g, float *out, cudaStream_t stream) {
  // Run kernel
  dim3 threads(32, 32);
  dim3 blocks((W+threads.x-1)/threads.x, (H+threads.y-1)/threads.y, C);
  ca_map_forward_kernel<<<blocks, threads, 0, stream>>>(weight, g, out, N, C, H, W);

  // Check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    return 0;
  else
    return 1;
}

extern "C" int _ca_map_backward_cuda(int N, int C, int H, int W, const float *dout, const float *weight, const float *g, float *dw, float *dg, cudaStream_t stream) {
  // Run kernel
  dim3 threads(32, 32);
  int d1 = (W+threads.x-1)/threads.x;
  int d2 = (H+threads.y-1)/threads.y;
  int d3 = H+W;
  dim3 blocks(d1, d2, d3);
  ca_map_backward_kernel_w<<<blocks, threads, 0, stream>>>(dout, weight, g, dw, N, C, H, W);

  d3 = C;
  blocks = dim3(d1, d2, d3);
  ca_map_backward_kernel_g<<<blocks, threads, 0, stream>>>(dout, weight, g, dg, N, C, H, W);

  // Check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    return 0;
  else
    return 1;
}

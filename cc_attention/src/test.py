__global__ void ca_forward_kernel(const float *t, const float *f, float *weight, int num, int chn, int height, int width) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int sp = height * width;
  int len = height + width - 1;
  int z = blockIdx.z;
  if (x < width && y < height && z < height + width - 1) {
	for (int batch = 0; batch < num; ++batch) {
		for (int plane = 0; plane < chn; ++plane) {
			float _t = 0;
			# 对于(x,y)点，将水平和垂直方向上(h+w-1)个点对应的通道值累加（方便计算每个点和其他所有点的关联）
			# 原本是依次水平和垂直方向上(h+w-1)个点和其他所有的点的关联，然后累加。这里先累加再做乘法
			for (int temp_x = 0; temp_x < width; ++temp_x) {
				_t += t[(batch * chn + plane) * sp + y * width + temp_x];
			}
			for (int temp_y = 0; temp_y < height; ++temp_y) {
				_t += t[(batch * chn + plane) * sp + temp_y * width + x];
			}
			if (z < width) {
				int i = z;
				float _f = f[(batch * chn + plane) * sp + y * width + i]; #取(x,y)关联的第i个点通道上的值
				weight[(batch * len + i) * sp + y * width + x] += _t * _f;
			}
			else {
				int i = z - width;
				int j = i < y ? i : i + 1;
				float _f = f[(batch * chn + plane) * sp + j * width + x];
				weight[(batch * len + width + i) * sp + y * width + x] += _t * _f;
			}
		}
	}
  }
}

# 原文：计算每个点和十字交叉方向点的关联，用softmax计算权重。

# CC-Pure-Python：
# 先计算h和h方向的关联，存放在通道维度。再计算w和w方向的关联，存放在通道维度。拼在一起，一起做softmax。

# 将十字方向的点作为邻域，计算邻域内每个点和其他所有点的关联，用softmax计算权重。这样关联大的点权重大。

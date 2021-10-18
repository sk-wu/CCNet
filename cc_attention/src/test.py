if (x < width && y < height && z < height + width - 1) {
	for (int batch = 0; batch < num; ++batch) {
		for (int plane = 0; plane < chn; ++plane) {
			float _t = 0;
			# 对于每个点，将水平和垂直方向上对应的通道值累加
			for (int temp_x = 0; temp_x < width; ++temp_x) {
				_t += t[(batch * chn + plane) * sp + y * width + temp_x];
			}
			for (int temp_y = 0; temp_y < height; ++temp_y) {
				_t += t[(batch * chn + plane) * sp + temp_y * width + x];
			}
			temp = 0;
			if (z < width) {
				int i = z;
				float _f = f[(batch * chn + plane) * sp + y * width + i];
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

# 原文：计算每个点和十字交叉方向点的关联，用softmax计算权重。

# CC-Pure-Python：
# 先计算h和h方向的关联，存放在通道维度。在计算w和w方向的关联，存放在通道维度。拼在一起，一起做softmax。

# 将十字方向的点作为邻域，计算邻域内每个点和其他所有点的关联，用softmax计算权重。这样关联大的点权重大。

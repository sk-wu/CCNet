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

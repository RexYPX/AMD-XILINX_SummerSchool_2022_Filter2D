#include <iostream>
using namespace std;
#include <math.h>
#include "filter2d.h"

#define N 128
#define K 3

void filter2d_accel(DTYPE* img_in, DTYPE* kernel, DTYPE* img_out, int rows, int cols)
{
#pragma HLS INTERFACE m_axi depth=128*128 port=img_in offset=slave bundle=axi_img_in
#pragma HLS INTERFACE m_axi depth=126*126 port=img_out bundle=axi_img_out
#pragma HLS INTERFACE m_axi depth=3*3 port=kernel offset=slave bundle=axi_kernel
#pragma HLS INTERFACE s_axilite port=rows  bundle=CTRL
#pragma HLS INTERFACE s_axilite port=cols  bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL

    uint16_t out_rows = rows - K + 1;
	uint16_t out_cols = cols - K + 1;
	uint16_t r,c,i,j;


    for(r=0; r<out_rows; r++)
    	for(c=0; c<out_cols; c++)
    	{
#pragma HLS PIPELINE
    		DTYPE* in_ptr = img_in + rows * r + c;
    		DTYPE* out_ptr = img_out + out_rows * r + c;
    		for(i=0; i<K; i++)
    			for(j=0; j<K; j++){
    				DTYPE img_temp = *(in_ptr + i*rows + j);
    				DTYPE filter_temp = *(kernel + i*K + j);
    				*(out_ptr) += img_temp * filter_temp;
    			}
    	}
}

#!/bin/bash
#set target device for XCLBIN
XDEVICE=xilinx:adm-pcie-ku3:2ddr:3.0

KERNEL_SRCS=krnls_cnn.cl
KERNEL_FUNC=
#KERNEL_CUs="--nk max_pool2:2 --nk relu_layer:2 --nk fully_connected_local:1 --nk conv_local_flatasync:1 --nk softmax_layer:1"

XCLBIN_NAME=bin_cnn_hw

if [ "$KERNEL_FUNC" == "" ]; then
	echo "No function selected!"
else
	$KERNEL_FUNC="-k $KERNEL_FUNC"
fi

xocc --xdevice ${XDEVICE} -t hw -o ${XCLBIN_NAME}.xclbin --report estimate  -j 8 -O3 ${KERNEL_SRCS}
#xocc --xdevice ${XDEVICE} -t hw -o ${XCLBIN_NAME}.xclbin ${KERNEL_CUs} --report estimate -j 4 -O3 ${KERNEL_SRCS}

#!/bin/bash
#set target device for XCLBIN
XDEVICE=xilinx:adm-pcie-ku3:2ddr:3.0

#set SDAccel build flow (sw_emu | hw_emu | hw)
SDA_FLOW=$1
if [ -z "$SDA_FLOW" ]; then
	echo "No SDA_FLOW passed, using 'hw'!"
	SDA_FLOW="hw"
fi

KERNEL_SRCS=krnls_cnn.cl
KERNEL_FUNC=
KERNEL_CUs="--nk conv2:2"

XCLBIN_NAME="bin_cnn_$SDA_FLOW"

if [ "$KERNEL_FUNC" == "" ]; then
	echo "No function selected!"
else
	$KERNEL_FUNC="-k $KERNEL_FUNC"
fi

echo "$SDA_FLOW build flow is choosen for ${KERNEL_SRCS}, output binary will be ${XCLBIN_NAME}.xclbin"
xocc --xdevice ${XDEVICE} -t ${SDA_FLOW} -o ${XCLBIN_NAME}.xclbin --report estimate -s -j 8 -O3 ${KERNEL_SRCS}
#xocc --xdevice ${XDEVICE} -t ${SDA_FLOW} -o ${XCLBIN_NAME}.xclbin ${KERNEL_CUs} --report estimate  -j 8 -O3 ${KERNEL_SRCS}


#ifndef __CNN_DESC__
#define __CNN_DESC__

// Convolution Layer 1
#define MASK1_SIZE 5
#define MASK1_DEPTH 1
#define FEAT1_OUT 32
#define IWIDTH1 28
#define OWIDTH1 24
#define IHEIGHT1 28
#define OHEIGHT1 24

// Convolution Layer 2
#define MASK2_SIZE 5
#define MASK2_DEPTH 32 
#define FEAT2_OUT 64
#define IWIDTH2 12
#define OWIDTH2 8
#define IHEIGHT2 12
#define OHEIGHT2 8

// FullyConnected Layer 1
#define INEURON1 1024
#define ONEURON1 256

// FullyConnected Layer 2
#define INEURON2 256
#define ONEURON2 10

#endif

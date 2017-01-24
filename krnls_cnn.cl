// DATA type and shape precisions
#define DATA_TYPE float
#define DATA_SHAPE uchar
//#define DATA_SHAPE size_t

// Helper function for relu layer after aggregation
DATA_TYPE relu(DATA_TYPE activation)
{
    return max((DATA_TYPE)0, activation);
}

// Helper function to generate index of a 2D mem block
size_t getIdx2D(const size_t y, const size_t x, const size_t width)
{
    return x + y * width;
}

// Helper function to generate index of a 3D mem block
size_t getIdx3D(const size_t z, const size_t y, const size_t x,
                            const size_t width, const size_t height)
{
    return x + y * (width) + z * (width * height);
}

// Max pooling layer with
// poolsize: 2x2
// stride equal to poolsize
// and 'SAME' padding policy
__kernel __attribute__((reqd_work_group_size(4, 4, 4)))
void max_pool2(__global DATA_TYPE * in, __global DATA_TYPE * out)
{
    size_t w = get_global_id(0);
    size_t h = get_global_id(1);
    size_t d = get_global_id(2);
    size_t owidth = get_global_size(0);
    size_t oheight = get_global_size(1);
    size_t iwidth = owidth * 2;
    size_t iheight = oheight * 2;

    size_t idx_tl = w*2 + iwidth * (h*2 + d * iheight);
    size_t idx_tr = w*2 + 1 + iwidth * (h*2 + d * iheight);
    size_t idx_bl = w*2 + iwidth * (h*2 + 1 + d * iheight);
    size_t idx_br = w*2 + 1 + iwidth * (h*2 + 1 + d * iheight);
    size_t out_idx = w + owidth * (h + d * oheight);

    out[out_idx] = max(in[idx_tl], max(in[idx_tr], max(in[idx_bl], in[idx_br])));
    return;
}

// Conv layer with
// kernel mask: 5x5
// stride: 1
// and 'VALID' padding policy
__kernel
void conv_layer(__global DATA_TYPE * in, __global DATA_TYPE * out,
            __constant DATA_TYPE * weight, __constant DATA_TYPE * biases,
            const DATA_SHAPE in_width, const DATA_SHAPE in_height,
            const DATA_SHAPE mask_depth)
{
    size_t w = get_global_id(0);
    size_t h = get_global_id(1);
    size_t d = get_global_id(2);
    size_t out_width = get_global_size(0);
    size_t out_height = get_global_size(1);

    size_t in_idx = w + in_width * (h + in_height * 0);

    DATA_TYPE c = (DATA_TYPE)0;
    for(size_t cd = 0; cd < mask_depth; ++cd)
    {
        for(size_t ch = 0; ch < 5; ++ch)
        {
            for(size_t cw = 0; cw < 5; ++cw)
            {
                c += in[in_idx + cw + (ch + cd * in_height) * in_width]
                * weight[cw + (ch + cd * mask_depth) * 5 + d * 5 * 5  * mask_depth];
            }
        }
    }
    size_t out_idx = w + out_width * (h + out_height * d);
    out[out_idx] = c + biases[d];
    return;
}

// Conv layer with local memory
// kernel mask: 5x5
// stride: 1
// and 'VALID' padding policy
#define CONV_WG_X 4
#define CONV_WG_Y 4
#define CONV_WG_Z 2

#define MASK_SIZE 5

#define TILE_X (CONV_WG_X+MASK_SIZE-1)
#define TILE_Y (CONV_WG_Y+MASK_SIZE-1)
#define TILE_Z 2

__kernel __attribute__((reqd_work_group_size(CONV_WG_X, CONV_WG_Y, CONV_WG_Z)))
void conv_local(__global DATA_TYPE * in, __global DATA_TYPE * out,
                __constant DATA_TYPE * weight, __constant DATA_TYPE * biases,
                const DATA_SHAPE in_width, const DATA_SHAPE in_height,
                const DATA_SHAPE mask_depth)
{
    __local DATA_TYPE tile[TILE_X * TILE_Y];

    size_t w = get_global_id(0);
    size_t h = get_global_id(1);
    size_t d = get_global_id(2);
    size_t out_width = get_global_size(0);
    size_t out_height = get_global_size(1);

    event_t event;
    size_t in_idx = get_group_id(0) * CONV_WG_X + get_group_id(1) * CONV_WG_Y * in_width;
    size_t out_idx = w + out_width * (h + out_height * d);

    DATA_TYPE c = (DATA_TYPE)0;
    for(size_t cd = 0; cd < mask_depth; ++cd)
    {
        for(size_t i = 0; i < TILE_Y; ++i)
            event = async_work_group_copy(&tile[i * TILE_X], &in[in_idx + i * in_width], TILE_X, event);
        in_idx += in_height * in_width;

        wait_group_events(1, &event);
        barrier(CLK_LOCAL_MEM_FENCE);

        for(size_t ch = 0; ch < MASK_SIZE; ++ch)
        {
            for(size_t cw = 0; cw < MASK_SIZE; ++cw)
            {
                c += tile[getIdx2D(ch + get_local_id(1), cw + get_local_id(0), TILE_X)]
                * weight[cw + (ch + cd * mask_depth) * MASK_SIZE + d * MASK_SIZE * MASK_SIZE * mask_depth];
            }
        }
    }
    out[out_idx] = relu(c + biases[d]);
    return;
}


// Fully connected layer
// kernel launch grid based on
// number of output neuron

#define INEURON 512 // num of input neuron
#define ONEURON 64  // num of output neuron for a work-group!
#define N_SYNAPSES (INEURON*ONEURON)

__kernel __attribute__((reqd_work_group_size(ONEURON, 1, 1)))
void fc_local(__global DATA_TYPE * in, __global DATA_TYPE * out,
                           __constant DATA_TYPE * weights, __constant DATA_TYPE * biases)
{
    size_t neuron = get_global_id(0);
    __local DATA_TYPE neuro_cache[INEURON];
    event_t event = async_work_group_copy(&neuro_cache[0], &in[0], INEURON, 0);

    wait_group_events(1, &event); 
    barrier(CLK_LOCAL_MEM_FENCE);

    DATA_TYPE n = 0;
    for(size_t c = 0; c < INEURON; ++c)
    {
        n += neuro_cache[c] * weights[neuron * INEURON + c];
    }
    out[neuron] = relu(n + biases[neuron]);
    return;
}


__kernel  __attribute__((reqd_work_group_size(1, 1, 1)))
void softmax_layer(__global DATA_TYPE * in, __global DATA_TYPE * out)
{
    DATA_TYPE soft[10];
    DATA_TYPE sum_exp = 0;

    for(uchar i = 0; i < 10; ++i)
    {
        soft[i] = native_exp(in[i]);
        sum_exp += soft[i];
    }

    for(uchar i = 0; i < 10; ++i)
    {
        out[i] = soft[i] / sum_exp;
    }
    return;
}



export SDACCEL_TIMELINE_REPORT=true
export SDACCEL_DEVICE_PROFILE=true

./cnn bin_cnn_hw.xclbin

unset SDACCEL_TIMELINE_REPORT
unset SDACCEL_DEVICE_PROFILE

sda2protobuf sdaccell_profile_summary.csv
sda2wdb sdaccel_timeline_trace.csv

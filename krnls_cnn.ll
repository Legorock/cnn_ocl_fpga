; ModuleID = 'krnls_cnn.cl'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"
target triple = "spir"

%opencl.event_t = type opaque

@conv_local_flatasync.tile = internal addrspace(3) global [128 x float] zeroinitializer, align 4
@conv_local_flatmem.tile = internal addrspace(3) global [64 x float] zeroinitializer, align 4
@fully_connected_local.neuro_cache = internal addrspace(3) global [512 x float] zeroinitializer, align 4

define spir_func float @_Z4reluf(float %activation) nounwind {
  %1 = alloca float, align 4
  store float %activation, float* %1, align 4
  %2 = load float* %1, align 4
  %3 = call spir_func float @_Z3maxff(float 0.000000e+00, float %2)
  ret float %3
}

declare spir_func float @_Z3maxff(float, float)

define spir_func i32 @_Z8getIdx2Djjj(i32 %y, i32 %x, i32 %width) nounwind {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  store i32 %y, i32* %1, align 4
  store i32 %x, i32* %2, align 4
  store i32 %width, i32* %3, align 4
  %4 = load i32* %2, align 4
  %5 = load i32* %1, align 4
  %6 = load i32* %3, align 4
  %7 = mul i32 %5, %6
  %8 = add i32 %4, %7
  ret i32 %8
}

define spir_func i32 @_Z8getIdx3Djjjjj(i32 %z, i32 %y, i32 %x, i32 %width, i32 %height) nounwind {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store i32 %z, i32* %1, align 4
  store i32 %y, i32* %2, align 4
  store i32 %x, i32* %3, align 4
  store i32 %width, i32* %4, align 4
  store i32 %height, i32* %5, align 4
  %6 = load i32* %3, align 4
  %7 = load i32* %2, align 4
  %8 = load i32* %4, align 4
  %9 = mul i32 %7, %8
  %10 = add i32 %6, %9
  %11 = load i32* %1, align 4
  %12 = load i32* %4, align 4
  %13 = load i32* %5, align 4
  %14 = mul i32 %12, %13
  %15 = mul i32 %11, %14
  %16 = add i32 %10, %15
  ret i32 %16
}

define spir_kernel void @max_pool2(float addrspace(1)* %in, float addrspace(1)* %out) nounwind {
  %1 = alloca float addrspace(1)*, align 4
  %2 = alloca float addrspace(1)*, align 4
  %w = alloca i32, align 4
  %h = alloca i32, align 4
  %d = alloca i32, align 4
  %owidth = alloca i32, align 4
  %oheight = alloca i32, align 4
  %iwidth = alloca i32, align 4
  %iheight = alloca i32, align 4
  %idx_tl = alloca i32, align 4
  %idx_tr = alloca i32, align 4
  %idx_bl = alloca i32, align 4
  %idx_br = alloca i32, align 4
  %out_idx = alloca i32, align 4
  store float addrspace(1)* %in, float addrspace(1)** %1, align 4
  store float addrspace(1)* %out, float addrspace(1)** %2, align 4
  %3 = call spir_func i32 @_Z13get_global_idj(i32 0)
  store i32 %3, i32* %w, align 4
  %4 = call spir_func i32 @_Z13get_global_idj(i32 1)
  store i32 %4, i32* %h, align 4
  %5 = call spir_func i32 @_Z13get_global_idj(i32 2)
  store i32 %5, i32* %d, align 4
  %6 = call spir_func i32 @_Z15get_global_sizej(i32 0)
  store i32 %6, i32* %owidth, align 4
  %7 = call spir_func i32 @_Z15get_global_sizej(i32 1)
  store i32 %7, i32* %oheight, align 4
  %8 = load i32* %owidth, align 4
  %9 = mul i32 %8, 2
  store i32 %9, i32* %iwidth, align 4
  %10 = load i32* %oheight, align 4
  %11 = mul i32 %10, 2
  store i32 %11, i32* %iheight, align 4
  %12 = load i32* %w, align 4
  %13 = mul i32 %12, 2
  %14 = load i32* %iwidth, align 4
  %15 = load i32* %h, align 4
  %16 = mul i32 %15, 2
  %17 = load i32* %d, align 4
  %18 = load i32* %iheight, align 4
  %19 = mul i32 %17, %18
  %20 = add i32 %16, %19
  %21 = mul i32 %14, %20
  %22 = add i32 %13, %21
  store i32 %22, i32* %idx_tl, align 4
  %23 = load i32* %w, align 4
  %24 = mul i32 %23, 2
  %25 = add i32 %24, 1
  %26 = load i32* %iwidth, align 4
  %27 = load i32* %h, align 4
  %28 = mul i32 %27, 2
  %29 = load i32* %d, align 4
  %30 = load i32* %iheight, align 4
  %31 = mul i32 %29, %30
  %32 = add i32 %28, %31
  %33 = mul i32 %26, %32
  %34 = add i32 %25, %33
  store i32 %34, i32* %idx_tr, align 4
  %35 = load i32* %w, align 4
  %36 = mul i32 %35, 2
  %37 = load i32* %iwidth, align 4
  %38 = load i32* %h, align 4
  %39 = mul i32 %38, 2
  %40 = add i32 %39, 1
  %41 = load i32* %d, align 4
  %42 = load i32* %iheight, align 4
  %43 = mul i32 %41, %42
  %44 = add i32 %40, %43
  %45 = mul i32 %37, %44
  %46 = add i32 %36, %45
  store i32 %46, i32* %idx_bl, align 4
  %47 = load i32* %w, align 4
  %48 = mul i32 %47, 2
  %49 = add i32 %48, 1
  %50 = load i32* %iwidth, align 4
  %51 = load i32* %h, align 4
  %52 = mul i32 %51, 2
  %53 = add i32 %52, 1
  %54 = load i32* %d, align 4
  %55 = load i32* %iheight, align 4
  %56 = mul i32 %54, %55
  %57 = add i32 %53, %56
  %58 = mul i32 %50, %57
  %59 = add i32 %49, %58
  store i32 %59, i32* %idx_br, align 4
  %60 = load i32* %w, align 4
  %61 = load i32* %owidth, align 4
  %62 = load i32* %h, align 4
  %63 = load i32* %d, align 4
  %64 = load i32* %oheight, align 4
  %65 = mul i32 %63, %64
  %66 = add i32 %62, %65
  %67 = mul i32 %61, %66
  %68 = add i32 %60, %67
  store i32 %68, i32* %out_idx, align 4
  %69 = load i32* %idx_tl, align 4
  %70 = load float addrspace(1)** %1, align 4
  %71 = getelementptr inbounds float addrspace(1)* %70, i32 %69
  %72 = load float addrspace(1)* %71, align 4
  %73 = load i32* %idx_tr, align 4
  %74 = load float addrspace(1)** %1, align 4
  %75 = getelementptr inbounds float addrspace(1)* %74, i32 %73
  %76 = load float addrspace(1)* %75, align 4
  %77 = load i32* %idx_bl, align 4
  %78 = load float addrspace(1)** %1, align 4
  %79 = getelementptr inbounds float addrspace(1)* %78, i32 %77
  %80 = load float addrspace(1)* %79, align 4
  %81 = load i32* %idx_br, align 4
  %82 = load float addrspace(1)** %1, align 4
  %83 = getelementptr inbounds float addrspace(1)* %82, i32 %81
  %84 = load float addrspace(1)* %83, align 4
  %85 = call spir_func float @_Z3maxff(float %80, float %84)
  %86 = call spir_func float @_Z3maxff(float %76, float %85)
  %87 = call spir_func float @_Z3maxff(float %72, float %86)
  %88 = load i32* %out_idx, align 4
  %89 = load float addrspace(1)** %2, align 4
  %90 = getelementptr inbounds float addrspace(1)* %89, i32 %88
  store float %87, float addrspace(1)* %90, align 4
  ret void
}

declare spir_func i32 @_Z13get_global_idj(i32)

declare spir_func i32 @_Z15get_global_sizej(i32)

define spir_kernel void @conv_layer(float addrspace(1)* %in, float addrspace(1)* %out, float addrspace(2)* %weight, float addrspace(2)* %biases, i8 zeroext %in_width, i8 zeroext %in_height, i8 zeroext %mask_depth) nounwind {
  %1 = alloca float addrspace(1)*, align 4
  %2 = alloca float addrspace(1)*, align 4
  %3 = alloca float addrspace(2)*, align 4
  %4 = alloca float addrspace(2)*, align 4
  %5 = alloca i8, align 1
  %6 = alloca i8, align 1
  %7 = alloca i8, align 1
  %w = alloca i32, align 4
  %h = alloca i32, align 4
  %d = alloca i32, align 4
  %out_width = alloca i32, align 4
  %out_height = alloca i32, align 4
  %in_idx = alloca i32, align 4
  %c = alloca float, align 4
  %cd = alloca i32, align 4
  %ch = alloca i32, align 4
  %cw = alloca i32, align 4
  %out_idx = alloca i32, align 4
  store float addrspace(1)* %in, float addrspace(1)** %1, align 4
  store float addrspace(1)* %out, float addrspace(1)** %2, align 4
  store float addrspace(2)* %weight, float addrspace(2)** %3, align 4
  store float addrspace(2)* %biases, float addrspace(2)** %4, align 4
  store i8 %in_width, i8* %5, align 1
  store i8 %in_height, i8* %6, align 1
  store i8 %mask_depth, i8* %7, align 1
  %8 = call spir_func i32 @_Z13get_global_idj(i32 0)
  store i32 %8, i32* %w, align 4
  %9 = call spir_func i32 @_Z13get_global_idj(i32 1)
  store i32 %9, i32* %h, align 4
  %10 = call spir_func i32 @_Z13get_global_idj(i32 2)
  store i32 %10, i32* %d, align 4
  %11 = call spir_func i32 @_Z15get_global_sizej(i32 0)
  store i32 %11, i32* %out_width, align 4
  %12 = call spir_func i32 @_Z15get_global_sizej(i32 1)
  store i32 %12, i32* %out_height, align 4
  %13 = load i32* %w, align 4
  %14 = load i8* %5, align 1
  %15 = zext i8 %14 to i32
  %16 = load i32* %h, align 4
  %17 = load i8* %6, align 1
  %18 = zext i8 %17 to i32
  %19 = mul nsw i32 %18, 0
  %20 = add i32 %16, %19
  %21 = mul i32 %15, %20
  %22 = add i32 %13, %21
  store i32 %22, i32* %in_idx, align 4
  store float 0.000000e+00, float* %c, align 4
  store i32 0, i32* %cd, align 4
  br label %23

; <label>:23                                      ; preds = %76, %0
  %24 = load i32* %cd, align 4
  %25 = load i8* %7, align 1
  %26 = zext i8 %25 to i32
  %27 = icmp ult i32 %24, %26
  br i1 %27, label %28, label %79

; <label>:28                                      ; preds = %23
  store i32 0, i32* %ch, align 4
  br label %29

; <label>:29                                      ; preds = %72, %28
  %30 = load i32* %ch, align 4
  %31 = icmp ult i32 %30, 5
  br i1 %31, label %32, label %75

; <label>:32                                      ; preds = %29
  store i32 0, i32* %cw, align 4
  br label %33

; <label>:33                                      ; preds = %68, %32
  %34 = load i32* %cw, align 4
  %35 = icmp ult i32 %34, 5
  br i1 %35, label %36, label %71

; <label>:36                                      ; preds = %33
  %37 = load i32* %in_idx, align 4
  %38 = load i32* %cw, align 4
  %39 = add i32 %37, %38
  %40 = load i32* %ch, align 4
  %41 = load i32* %cd, align 4
  %42 = load i8* %6, align 1
  %43 = zext i8 %42 to i32
  %44 = mul i32 %41, %43
  %45 = add i32 %40, %44
  %46 = load i8* %5, align 1
  %47 = zext i8 %46 to i32
  %48 = mul i32 %45, %47
  %49 = add i32 %39, %48
  %50 = load float addrspace(1)** %1, align 4
  %51 = getelementptr inbounds float addrspace(1)* %50, i32 %49
  %52 = load float addrspace(1)* %51, align 4
  %53 = load i32* %cw, align 4
  %54 = load i32* %ch, align 4
  %55 = load i32* %cd, align 4
  %56 = load i8* %7, align 1
  %57 = zext i8 %56 to i32
  %58 = mul i32 %55, %57
  %59 = add i32 %54, %58
  %60 = mul i32 %59, 5
  %61 = add i32 %53, %60
  %62 = load float addrspace(2)** %3, align 4
  %63 = getelementptr inbounds float addrspace(2)* %62, i32 %61
  %64 = load float addrspace(2)* %63, align 4
  %65 = fmul float %52, %64
  %66 = load float* %c, align 4
  %67 = fadd float %66, %65
  store float %67, float* %c, align 4
  br label %68

; <label>:68                                      ; preds = %36
  %69 = load i32* %cw, align 4
  %70 = add i32 %69, 1
  store i32 %70, i32* %cw, align 4
  br label %33

; <label>:71                                      ; preds = %33
  br label %72

; <label>:72                                      ; preds = %71
  %73 = load i32* %ch, align 4
  %74 = add i32 %73, 1
  store i32 %74, i32* %ch, align 4
  br label %29

; <label>:75                                      ; preds = %29
  br label %76

; <label>:76                                      ; preds = %75
  %77 = load i32* %cd, align 4
  %78 = add i32 %77, 1
  store i32 %78, i32* %cd, align 4
  br label %23

; <label>:79                                      ; preds = %23
  %80 = load i32* %w, align 4
  %81 = load i32* %out_width, align 4
  %82 = load i32* %h, align 4
  %83 = load i32* %out_height, align 4
  %84 = load i32* %d, align 4
  %85 = mul i32 %83, %84
  %86 = add i32 %82, %85
  %87 = mul i32 %81, %86
  %88 = add i32 %80, %87
  store i32 %88, i32* %out_idx, align 4
  %89 = load float* %c, align 4
  %90 = load i32* %d, align 4
  %91 = load float addrspace(2)** %4, align 4
  %92 = getelementptr inbounds float addrspace(2)* %91, i32 %90
  %93 = load float addrspace(2)* %92, align 4
  %94 = fadd float %89, %93
  %95 = load i32* %out_idx, align 4
  %96 = load float addrspace(1)** %2, align 4
  %97 = getelementptr inbounds float addrspace(1)* %96, i32 %95
  store float %94, float addrspace(1)* %97, align 4
  ret void
}

define spir_kernel void @conv_local_flatasync(float addrspace(1)* %in, float addrspace(1)* %out, float addrspace(2)* %weight, float addrspace(2)* %biases, i8 zeroext %in_width, i8 zeroext %in_height, i8 zeroext %mask_depth, i8 zeroext %mask_size) nounwind {
  %1 = alloca float addrspace(1)*, align 4
  %2 = alloca float addrspace(1)*, align 4
  %3 = alloca float addrspace(2)*, align 4
  %4 = alloca float addrspace(2)*, align 4
  %5 = alloca i8, align 1
  %6 = alloca i8, align 1
  %7 = alloca i8, align 1
  %8 = alloca i8, align 1
  %w = alloca i32, align 4
  %h = alloca i32, align 4
  %d = alloca i32, align 4
  %out_width = alloca i32, align 4
  %out_height = alloca i32, align 4
  %events = alloca [2 x %opencl.event_t*], align 4
  %e = alloca i32, align 4
  %in_idx = alloca i32, align 4
  %out_idx = alloca i32, align 4
  %i = alloca i32, align 4
  %c = alloca float, align 4
  %cd = alloca i32, align 4
  %ch = alloca i32, align 4
  %cw = alloca i32, align 4
  %i1 = alloca i32, align 4
  store float addrspace(1)* %in, float addrspace(1)** %1, align 4
  store float addrspace(1)* %out, float addrspace(1)** %2, align 4
  store float addrspace(2)* %weight, float addrspace(2)** %3, align 4
  store float addrspace(2)* %biases, float addrspace(2)** %4, align 4
  store i8 %in_width, i8* %5, align 1
  store i8 %in_height, i8* %6, align 1
  store i8 %mask_depth, i8* %7, align 1
  store i8 %mask_size, i8* %8, align 1
  %9 = call spir_func i32 @_Z13get_global_idj(i32 0)
  store i32 %9, i32* %w, align 4
  %10 = call spir_func i32 @_Z13get_global_idj(i32 1)
  store i32 %10, i32* %h, align 4
  %11 = call spir_func i32 @_Z13get_global_idj(i32 2)
  store i32 %11, i32* %d, align 4
  %12 = call spir_func i32 @_Z15get_global_sizej(i32 0)
  store i32 %12, i32* %out_width, align 4
  %13 = call spir_func i32 @_Z15get_global_sizej(i32 1)
  store i32 %13, i32* %out_height, align 4
  store i32 0, i32* %e, align 4
  %14 = call spir_func i32 @_Z12get_group_idj(i32 0)
  %15 = mul i32 %14, 4
  %16 = call spir_func i32 @_Z12get_group_idj(i32 1)
  %17 = mul i32 %16, 4
  %18 = load i8* %5, align 1
  %19 = zext i8 %18 to i32
  %20 = mul i32 %17, %19
  %21 = add i32 %15, %20
  store i32 %21, i32* %in_idx, align 4
  %22 = load i32* %w, align 4
  %23 = load i32* %out_width, align 4
  %24 = load i32* %h, align 4
  %25 = load i32* %out_height, align 4
  %26 = load i32* %d, align 4
  %27 = mul i32 %25, %26
  %28 = add i32 %24, %27
  %29 = mul i32 %23, %28
  %30 = add i32 %22, %29
  store i32 %30, i32* %out_idx, align 4
  store i32 0, i32* %i, align 4
  br label %31

; <label>:31                                      ; preds = %53, %0
  %32 = load i32* %i, align 4
  %33 = icmp ult i32 %32, 8
  br i1 %33, label %34, label %56

; <label>:34                                      ; preds = %31
  %35 = load i32* %e, align 4
  %36 = load i32* %i, align 4
  %37 = call spir_func i32 @_Z8getIdx3Djjjjj(i32 %35, i32 %36, i32 0, i32 8, i32 8)
  %38 = getelementptr inbounds [128 x float] addrspace(3)* @conv_local_flatasync.tile, i32 0, i32 %37
  %39 = load i32* %in_idx, align 4
  %40 = load i32* %i, align 4
  %41 = load i8* %5, align 1
  %42 = zext i8 %41 to i32
  %43 = mul i32 %40, %42
  %44 = add i32 %39, %43
  %45 = load float addrspace(1)** %1, align 4
  %46 = getelementptr inbounds float addrspace(1)* %45, i32 %44
  %47 = load i32* %e, align 4
  %48 = getelementptr inbounds [2 x %opencl.event_t*]* %events, i32 0, i32 %47
  %49 = load %opencl.event_t** %48, align 4
  %50 = call spir_func %opencl.event_t* @_Z21async_work_group_copyPU3AS1fPKU3AS0fj9ocl_event(float addrspace(3)* %38, float addrspace(1)* %46, i32 8, %opencl.event_t* %49)
  %51 = load i32* %e, align 4
  %52 = getelementptr inbounds [2 x %opencl.event_t*]* %events, i32 0, i32 %51
  store %opencl.event_t* %50, %opencl.event_t** %52, align 4
  br label %53

; <label>:53                                      ; preds = %34
  %54 = load i32* %i, align 4
  %55 = add i32 %54, 1
  store i32 %55, i32* %i, align 4
  br label %31

; <label>:56                                      ; preds = %31
  store float 0.000000e+00, float* %c, align 4
  store i32 0, i32* %cd, align 4
  br label %57

; <label>:57                                      ; preds = %149, %56
  %58 = load i32* %cd, align 4
  %59 = load i8* %7, align 1
  %60 = zext i8 %59 to i32
  %61 = icmp ult i32 %58, %60
  br i1 %61, label %62, label %152

; <label>:62                                      ; preds = %57
  %63 = load i32* %e, align 4
  %64 = getelementptr inbounds [2 x %opencl.event_t*]* %events, i32 0, i32 %63
  call spir_func void @_Z17wait_group_eventsiP9ocl_event(i32 1, %opencl.event_t** %64)
  call spir_func void @_Z7barrierj(i32 0)
  store i32 0, i32* %ch, align 4
  br label %65

; <label>:65                                      ; preds = %102, %62
  %66 = load i32* %ch, align 4
  %67 = icmp ult i32 %66, 5
  br i1 %67, label %68, label %105

; <label>:68                                      ; preds = %65
  store i32 0, i32* %cw, align 4
  br label %69

; <label>:69                                      ; preds = %98, %68
  %70 = load i32* %cw, align 4
  %71 = icmp ult i32 %70, 5
  br i1 %71, label %72, label %101

; <label>:72                                      ; preds = %69
  %73 = load i32* %e, align 4
  %74 = load i32* %ch, align 4
  %75 = call spir_func i32 @_Z12get_local_idj(i32 1)
  %76 = add i32 %74, %75
  %77 = load i32* %cw, align 4
  %78 = call spir_func i32 @_Z12get_local_idj(i32 0)
  %79 = add i32 %77, %78
  %80 = call spir_func i32 @_Z8getIdx3Djjjjj(i32 %73, i32 %76, i32 %79, i32 8, i32 8)
  %81 = getelementptr inbounds [128 x float] addrspace(3)* @conv_local_flatasync.tile, i32 0, i32 %80
  %82 = load float addrspace(3)* %81, align 4
  %83 = load i32* %cw, align 4
  %84 = load i32* %ch, align 4
  %85 = load i32* %cd, align 4
  %86 = load i8* %7, align 1
  %87 = zext i8 %86 to i32
  %88 = mul i32 %85, %87
  %89 = add i32 %84, %88
  %90 = mul i32 %89, 5
  %91 = add i32 %83, %90
  %92 = load float addrspace(2)** %3, align 4
  %93 = getelementptr inbounds float addrspace(2)* %92, i32 %91
  %94 = load float addrspace(2)* %93, align 4
  %95 = fmul float %82, %94
  %96 = load float* %c, align 4
  %97 = fadd float %96, %95
  store float %97, float* %c, align 4
  br label %98

; <label>:98                                      ; preds = %72
  %99 = load i32* %cw, align 4
  %100 = add i32 %99, 1
  store i32 %100, i32* %cw, align 4
  br label %69

; <label>:101                                     ; preds = %69
  br label %102

; <label>:102                                     ; preds = %101
  %103 = load i32* %ch, align 4
  %104 = add i32 %103, 1
  store i32 %104, i32* %ch, align 4
  br label %65

; <label>:105                                     ; preds = %65
  %106 = load i32* %e, align 4
  %107 = icmp eq i32 %106, 0
  %108 = select i1 %107, i32 1, i32 0
  store i32 %108, i32* %e, align 4
  %109 = load i32* %cd, align 4
  %110 = load i8* %7, align 1
  %111 = zext i8 %110 to i32
  %112 = sub nsw i32 %111, 1
  %113 = icmp ult i32 %109, %112
  br i1 %113, label %114, label %148

; <label>:114                                     ; preds = %105
  %115 = load i8* %6, align 1
  %116 = zext i8 %115 to i32
  %117 = load i8* %5, align 1
  %118 = zext i8 %117 to i32
  %119 = mul nsw i32 %116, %118
  %120 = load i32* %in_idx, align 4
  %121 = add i32 %120, %119
  store i32 %121, i32* %in_idx, align 4
  store i32 0, i32* %i1, align 4
  br label %122

; <label>:122                                     ; preds = %144, %114
  %123 = load i32* %i1, align 4
  %124 = icmp ult i32 %123, 8
  br i1 %124, label %125, label %147

; <label>:125                                     ; preds = %122
  %126 = load i32* %e, align 4
  %127 = load i32* %i1, align 4
  %128 = call spir_func i32 @_Z8getIdx3Djjjjj(i32 %126, i32 %127, i32 0, i32 8, i32 8)
  %129 = getelementptr inbounds [128 x float] addrspace(3)* @conv_local_flatasync.tile, i32 0, i32 %128
  %130 = load i32* %in_idx, align 4
  %131 = load i32* %i1, align 4
  %132 = load i8* %5, align 1
  %133 = zext i8 %132 to i32
  %134 = mul i32 %131, %133
  %135 = add i32 %130, %134
  %136 = load float addrspace(1)** %1, align 4
  %137 = getelementptr inbounds float addrspace(1)* %136, i32 %135
  %138 = load i32* %e, align 4
  %139 = getelementptr inbounds [2 x %opencl.event_t*]* %events, i32 0, i32 %138
  %140 = load %opencl.event_t** %139, align 4
  %141 = call spir_func %opencl.event_t* @_Z21async_work_group_copyPU3AS1fPKU3AS0fj9ocl_event(float addrspace(3)* %129, float addrspace(1)* %137, i32 8, %opencl.event_t* %140)
  %142 = load i32* %e, align 4
  %143 = getelementptr inbounds [2 x %opencl.event_t*]* %events, i32 0, i32 %142
  store %opencl.event_t* %141, %opencl.event_t** %143, align 4
  br label %144

; <label>:144                                     ; preds = %125
  %145 = load i32* %i1, align 4
  %146 = add i32 %145, 1
  store i32 %146, i32* %i1, align 4
  br label %122

; <label>:147                                     ; preds = %122
  br label %148

; <label>:148                                     ; preds = %147, %105
  br label %149

; <label>:149                                     ; preds = %148
  %150 = load i32* %cd, align 4
  %151 = add i32 %150, 1
  store i32 %151, i32* %cd, align 4
  br label %57

; <label>:152                                     ; preds = %57
  %153 = load float* %c, align 4
  %154 = load i32* %d, align 4
  %155 = load float addrspace(2)** %4, align 4
  %156 = getelementptr inbounds float addrspace(2)* %155, i32 %154
  %157 = load float addrspace(2)* %156, align 4
  %158 = fadd float %153, %157
  %159 = call spir_func float @_Z4reluf(float %158)
  %160 = load i32* %out_idx, align 4
  %161 = load float addrspace(1)** %2, align 4
  %162 = getelementptr inbounds float addrspace(1)* %161, i32 %160
  store float %159, float addrspace(1)* %162, align 4
  ret void
}

declare spir_func i32 @_Z12get_group_idj(i32)

declare spir_func %opencl.event_t* @_Z21async_work_group_copyPU3AS1fPKU3AS0fj9ocl_event(float addrspace(3)*, float addrspace(1)*, i32, %opencl.event_t*)

declare spir_func void @_Z17wait_group_eventsiP9ocl_event(i32, %opencl.event_t**)

declare spir_func void @_Z7barrierj(i32)

declare spir_func i32 @_Z12get_local_idj(i32)

define spir_kernel void @conv_local_flatmem(float addrspace(1)* %in, float addrspace(1)* %out, float addrspace(2)* %weight, float addrspace(2)* %biases, i8 zeroext %in_width, i8 zeroext %in_height, i8 zeroext %mask_depth) nounwind {
  %1 = alloca float addrspace(1)*, align 4
  %2 = alloca float addrspace(1)*, align 4
  %3 = alloca float addrspace(2)*, align 4
  %4 = alloca float addrspace(2)*, align 4
  %5 = alloca i8, align 1
  %6 = alloca i8, align 1
  %7 = alloca i8, align 1
  %w = alloca i32, align 4
  %h = alloca i32, align 4
  %d = alloca i32, align 4
  %out_width = alloca i32, align 4
  %out_height = alloca i32, align 4
  %event = alloca %opencl.event_t*, align 4
  %in_idx = alloca i32, align 4
  %out_idx = alloca i32, align 4
  %c = alloca float, align 4
  %cd = alloca i32, align 4
  %i = alloca i32, align 4
  %ch = alloca i32, align 4
  %cw = alloca i32, align 4
  store float addrspace(1)* %in, float addrspace(1)** %1, align 4
  store float addrspace(1)* %out, float addrspace(1)** %2, align 4
  store float addrspace(2)* %weight, float addrspace(2)** %3, align 4
  store float addrspace(2)* %biases, float addrspace(2)** %4, align 4
  store i8 %in_width, i8* %5, align 1
  store i8 %in_height, i8* %6, align 1
  store i8 %mask_depth, i8* %7, align 1
  %8 = call spir_func i32 @_Z13get_global_idj(i32 0)
  store i32 %8, i32* %w, align 4
  %9 = call spir_func i32 @_Z13get_global_idj(i32 1)
  store i32 %9, i32* %h, align 4
  %10 = call spir_func i32 @_Z13get_global_idj(i32 2)
  store i32 %10, i32* %d, align 4
  %11 = call spir_func i32 @_Z15get_global_sizej(i32 0)
  store i32 %11, i32* %out_width, align 4
  %12 = call spir_func i32 @_Z15get_global_sizej(i32 1)
  store i32 %12, i32* %out_height, align 4
  %13 = call spir_func i32 @_Z12get_group_idj(i32 0)
  %14 = mul i32 %13, 4
  %15 = call spir_func i32 @_Z12get_group_idj(i32 1)
  %16 = mul i32 %15, 4
  %17 = load i8* %5, align 1
  %18 = zext i8 %17 to i32
  %19 = mul i32 %16, %18
  %20 = add i32 %14, %19
  store i32 %20, i32* %in_idx, align 4
  %21 = load i32* %w, align 4
  %22 = load i32* %out_width, align 4
  %23 = load i32* %h, align 4
  %24 = load i32* %out_height, align 4
  %25 = load i32* %d, align 4
  %26 = mul i32 %24, %25
  %27 = add i32 %23, %26
  %28 = mul i32 %22, %27
  %29 = add i32 %21, %28
  store i32 %29, i32* %out_idx, align 4
  store float 0.000000e+00, float* %c, align 4
  store i32 0, i32* %cd, align 4
  br label %30

; <label>:30                                      ; preds = %104, %0
  %31 = load i32* %cd, align 4
  %32 = load i8* %7, align 1
  %33 = zext i8 %32 to i32
  %34 = icmp ult i32 %31, %33
  br i1 %34, label %35, label %107

; <label>:35                                      ; preds = %30
  store i32 0, i32* %i, align 4
  br label %36

; <label>:36                                      ; preds = %53, %35
  %37 = load i32* %i, align 4
  %38 = icmp ult i32 %37, 8
  br i1 %38, label %39, label %56

; <label>:39                                      ; preds = %36
  %40 = load i32* %i, align 4
  %41 = mul i32 %40, 8
  %42 = getelementptr inbounds [64 x float] addrspace(3)* @conv_local_flatmem.tile, i32 0, i32 %41
  %43 = load i32* %in_idx, align 4
  %44 = load i32* %i, align 4
  %45 = load i8* %5, align 1
  %46 = zext i8 %45 to i32
  %47 = mul i32 %44, %46
  %48 = add i32 %43, %47
  %49 = load float addrspace(1)** %1, align 4
  %50 = getelementptr inbounds float addrspace(1)* %49, i32 %48
  %51 = load %opencl.event_t** %event, align 4
  %52 = call spir_func %opencl.event_t* @_Z21async_work_group_copyPU3AS1fPKU3AS0fj9ocl_event(float addrspace(3)* %42, float addrspace(1)* %50, i32 8, %opencl.event_t* %51)
  store %opencl.event_t* %52, %opencl.event_t** %event, align 4
  br label %53

; <label>:53                                      ; preds = %39
  %54 = load i32* %i, align 4
  %55 = add i32 %54, 1
  store i32 %55, i32* %i, align 4
  br label %36

; <label>:56                                      ; preds = %36
  %57 = load i8* %6, align 1
  %58 = zext i8 %57 to i32
  %59 = load i8* %5, align 1
  %60 = zext i8 %59 to i32
  %61 = mul nsw i32 %58, %60
  %62 = load i32* %in_idx, align 4
  %63 = add i32 %62, %61
  store i32 %63, i32* %in_idx, align 4
  call spir_func void @_Z17wait_group_eventsiP9ocl_event(i32 1, %opencl.event_t** %event)
  call spir_func void @_Z7barrierj(i32 0)
  store i32 0, i32* %ch, align 4
  br label %64

; <label>:64                                      ; preds = %100, %56
  %65 = load i32* %ch, align 4
  %66 = icmp ult i32 %65, 5
  br i1 %66, label %67, label %103

; <label>:67                                      ; preds = %64
  store i32 0, i32* %cw, align 4
  br label %68

; <label>:68                                      ; preds = %96, %67
  %69 = load i32* %cw, align 4
  %70 = icmp ult i32 %69, 5
  br i1 %70, label %71, label %99

; <label>:71                                      ; preds = %68
  %72 = load i32* %ch, align 4
  %73 = call spir_func i32 @_Z12get_local_idj(i32 1)
  %74 = add i32 %72, %73
  %75 = load i32* %cw, align 4
  %76 = call spir_func i32 @_Z12get_local_idj(i32 0)
  %77 = add i32 %75, %76
  %78 = call spir_func i32 @_Z8getIdx2Djjj(i32 %74, i32 %77, i32 8)
  %79 = getelementptr inbounds [64 x float] addrspace(3)* @conv_local_flatmem.tile, i32 0, i32 %78
  %80 = load float addrspace(3)* %79, align 4
  %81 = load i32* %cw, align 4
  %82 = load i32* %ch, align 4
  %83 = load i32* %cd, align 4
  %84 = load i8* %7, align 1
  %85 = zext i8 %84 to i32
  %86 = mul i32 %83, %85
  %87 = add i32 %82, %86
  %88 = mul i32 %87, 5
  %89 = add i32 %81, %88
  %90 = load float addrspace(2)** %3, align 4
  %91 = getelementptr inbounds float addrspace(2)* %90, i32 %89
  %92 = load float addrspace(2)* %91, align 4
  %93 = fmul float %80, %92
  %94 = load float* %c, align 4
  %95 = fadd float %94, %93
  store float %95, float* %c, align 4
  br label %96

; <label>:96                                      ; preds = %71
  %97 = load i32* %cw, align 4
  %98 = add i32 %97, 1
  store i32 %98, i32* %cw, align 4
  br label %68

; <label>:99                                      ; preds = %68
  br label %100

; <label>:100                                     ; preds = %99
  %101 = load i32* %ch, align 4
  %102 = add i32 %101, 1
  store i32 %102, i32* %ch, align 4
  br label %64

; <label>:103                                     ; preds = %64
  br label %104

; <label>:104                                     ; preds = %103
  %105 = load i32* %cd, align 4
  %106 = add i32 %105, 1
  store i32 %106, i32* %cd, align 4
  br label %30

; <label>:107                                     ; preds = %30
  %108 = load float* %c, align 4
  %109 = load i32* %d, align 4
  %110 = load float addrspace(2)** %4, align 4
  %111 = getelementptr inbounds float addrspace(2)* %110, i32 %109
  %112 = load float addrspace(2)* %111, align 4
  %113 = fadd float %108, %112
  %114 = call spir_func float @_Z4reluf(float %113)
  %115 = load i32* %out_idx, align 4
  %116 = load float addrspace(1)** %2, align 4
  %117 = getelementptr inbounds float addrspace(1)* %116, i32 %115
  store float %114, float addrspace(1)* %117, align 4
  ret void
}

define spir_kernel void @fully_connected_local(float addrspace(1)* %in, float addrspace(1)* %out, float addrspace(2)* %weights, float addrspace(2)* %biases) nounwind {
  %1 = alloca float addrspace(1)*, align 4
  %2 = alloca float addrspace(1)*, align 4
  %3 = alloca float addrspace(2)*, align 4
  %4 = alloca float addrspace(2)*, align 4
  %neuron = alloca i32, align 4
  %event = alloca %opencl.event_t*, align 4
  %n = alloca float, align 4
  %c = alloca i32, align 4
  store float addrspace(1)* %in, float addrspace(1)** %1, align 4
  store float addrspace(1)* %out, float addrspace(1)** %2, align 4
  store float addrspace(2)* %weights, float addrspace(2)** %3, align 4
  store float addrspace(2)* %biases, float addrspace(2)** %4, align 4
  %5 = call spir_func i32 @_Z13get_global_idj(i32 0)
  store i32 %5, i32* %neuron, align 4
  %6 = load float addrspace(1)** %1, align 4
  %7 = getelementptr inbounds float addrspace(1)* %6, i32 0
  %8 = call spir_func %opencl.event_t* @_Z21async_work_group_copyPU3AS1fPKU3AS0fj9ocl_event(float addrspace(3)* getelementptr inbounds ([512 x float] addrspace(3)* @fully_connected_local.neuro_cache, i32 0, i32 0), float addrspace(1)* %7, i32 512, %opencl.event_t* null)
  store %opencl.event_t* %8, %opencl.event_t** %event, align 4
  call spir_func void @_Z17wait_group_eventsiP9ocl_event(i32 1, %opencl.event_t** %event)
  store float 0.000000e+00, float* %n, align 4
  store i32 0, i32* %c, align 4
  br label %9

; <label>:9                                       ; preds = %26, %0
  %10 = load i32* %c, align 4
  %11 = icmp ult i32 %10, 512
  br i1 %11, label %12, label %29

; <label>:12                                      ; preds = %9
  %13 = load i32* %c, align 4
  %14 = getelementptr inbounds [512 x float] addrspace(3)* @fully_connected_local.neuro_cache, i32 0, i32 %13
  %15 = load float addrspace(3)* %14, align 4
  %16 = load i32* %neuron, align 4
  %17 = mul i32 %16, 512
  %18 = load i32* %c, align 4
  %19 = add i32 %17, %18
  %20 = load float addrspace(2)** %3, align 4
  %21 = getelementptr inbounds float addrspace(2)* %20, i32 %19
  %22 = load float addrspace(2)* %21, align 4
  %23 = fmul float %15, %22
  %24 = load float* %n, align 4
  %25 = fadd float %24, %23
  store float %25, float* %n, align 4
  br label %26

; <label>:26                                      ; preds = %12
  %27 = load i32* %c, align 4
  %28 = add i32 %27, 1
  store i32 %28, i32* %c, align 4
  br label %9

; <label>:29                                      ; preds = %9
  %30 = load float* %n, align 4
  %31 = load i32* %neuron, align 4
  %32 = load float addrspace(2)** %4, align 4
  %33 = getelementptr inbounds float addrspace(2)* %32, i32 %31
  %34 = load float addrspace(2)* %33, align 4
  %35 = fadd float %30, %34
  %36 = call spir_func float @_Z4reluf(float %35)
  %37 = load i32* %neuron, align 4
  %38 = load float addrspace(1)** %2, align 4
  %39 = getelementptr inbounds float addrspace(1)* %38, i32 %37
  store float %36, float addrspace(1)* %39, align 4
  ret void
}

define spir_kernel void @softmax_layer(float addrspace(1)* %in, float addrspace(1)* %out) nounwind {
  %1 = alloca float addrspace(1)*, align 4
  %2 = alloca float addrspace(1)*, align 4
  %soft = alloca [10 x float], align 4
  %sum_exp = alloca float, align 4
  %i = alloca i8, align 1
  %i1 = alloca i8, align 1
  store float addrspace(1)* %in, float addrspace(1)** %1, align 4
  store float addrspace(1)* %out, float addrspace(1)** %2, align 4
  store float 0.000000e+00, float* %sum_exp, align 4
  store i8 0, i8* %i, align 1
  br label %3

; <label>:3                                       ; preds = %23, %0
  %4 = load i8* %i, align 1
  %5 = zext i8 %4 to i32
  %6 = icmp slt i32 %5, 10
  br i1 %6, label %7, label %26

; <label>:7                                       ; preds = %3
  %8 = load i8* %i, align 1
  %9 = zext i8 %8 to i32
  %10 = load float addrspace(1)** %1, align 4
  %11 = getelementptr inbounds float addrspace(1)* %10, i32 %9
  %12 = load float addrspace(1)* %11, align 4
  %13 = call spir_func float @_Z10native_expf(float %12)
  %14 = load i8* %i, align 1
  %15 = zext i8 %14 to i32
  %16 = getelementptr inbounds [10 x float]* %soft, i32 0, i32 %15
  store float %13, float* %16, align 4
  %17 = load i8* %i, align 1
  %18 = zext i8 %17 to i32
  %19 = getelementptr inbounds [10 x float]* %soft, i32 0, i32 %18
  %20 = load float* %19, align 4
  %21 = load float* %sum_exp, align 4
  %22 = fadd float %21, %20
  store float %22, float* %sum_exp, align 4
  br label %23

; <label>:23                                      ; preds = %7
  %24 = load i8* %i, align 1
  %25 = add i8 %24, 1
  store i8 %25, i8* %i, align 1
  br label %3

; <label>:26                                      ; preds = %3
  store i8 0, i8* %i1, align 1
  br label %27

; <label>:27                                      ; preds = %42, %26
  %28 = load i8* %i1, align 1
  %29 = zext i8 %28 to i32
  %30 = icmp slt i32 %29, 10
  br i1 %30, label %31, label %45

; <label>:31                                      ; preds = %27
  %32 = load i8* %i1, align 1
  %33 = zext i8 %32 to i32
  %34 = getelementptr inbounds [10 x float]* %soft, i32 0, i32 %33
  %35 = load float* %34, align 4
  %36 = load float* %sum_exp, align 4
  %37 = fdiv float %35, %36, !fpmath !30
  %38 = load i8* %i1, align 1
  %39 = zext i8 %38 to i32
  %40 = load float addrspace(1)** %2, align 4
  %41 = getelementptr inbounds float addrspace(1)* %40, i32 %39
  store float %37, float addrspace(1)* %41, align 4
  br label %42

; <label>:42                                      ; preds = %31
  %43 = load i8* %i1, align 1
  %44 = add i8 %43, 1
  store i8 %44, i8* %i1, align 1
  br label %27

; <label>:45                                      ; preds = %27
  ret void
}

declare spir_func float @_Z10native_expf(float)

!opencl.kernels = !{!0, !7, !13, !20, !21, !28}

!0 = metadata !{void (float addrspace(1)*, float addrspace(1)*)* @max_pool2, metadata !1, metadata !2, metadata !3, metadata !4, metadata !5, metadata !6}
!1 = metadata !{metadata !"kernel_arg_addr_space", i32 1, i32 1}
!2 = metadata !{metadata !"kernel_arg_access_qual", metadata !"none", metadata !"none"}
!3 = metadata !{metadata !"kernel_arg_type", metadata !"float*", metadata !"float*"}
!4 = metadata !{metadata !"kernel_arg_type_qual", metadata !"", metadata !""}
!5 = metadata !{metadata !"kernel_arg_name", metadata !"in", metadata !"out"}
!6 = metadata !{metadata !"reqd_work_group_size", i32 4, i32 4, i32 4}
!7 = metadata !{void (float addrspace(1)*, float addrspace(1)*, float addrspace(2)*, float addrspace(2)*, i8, i8, i8)* @conv_layer, metadata !8, metadata !9, metadata !10, metadata !11, metadata !12}
!8 = metadata !{metadata !"kernel_arg_addr_space", i32 1, i32 1, i32 2, i32 2, i32 0, i32 0, i32 0}
!9 = metadata !{metadata !"kernel_arg_access_qual", metadata !"none", metadata !"none", metadata !"none", metadata !"none", metadata !"none", metadata !"none", metadata !"none"}
!10 = metadata !{metadata !"kernel_arg_type", metadata !"float*", metadata !"float*", metadata !"float*", metadata !"float*", metadata !"uchar", metadata !"uchar", metadata !"uchar"}
!11 = metadata !{metadata !"kernel_arg_type_qual", metadata !"", metadata !"", metadata !"const", metadata !"const", metadata !"const", metadata !"const", metadata !"const"}
!12 = metadata !{metadata !"kernel_arg_name", metadata !"in", metadata !"out", metadata !"weight", metadata !"biases", metadata !"in_width", metadata !"in_height", metadata !"mask_depth"}
!13 = metadata !{void (float addrspace(1)*, float addrspace(1)*, float addrspace(2)*, float addrspace(2)*, i8, i8, i8, i8)* @conv_local_flatasync, metadata !14, metadata !15, metadata !16, metadata !17, metadata !18, metadata !19}
!14 = metadata !{metadata !"kernel_arg_addr_space", i32 1, i32 1, i32 2, i32 2, i32 0, i32 0, i32 0, i32 0}
!15 = metadata !{metadata !"kernel_arg_access_qual", metadata !"none", metadata !"none", metadata !"none", metadata !"none", metadata !"none", metadata !"none", metadata !"none", metadata !"none"}
!16 = metadata !{metadata !"kernel_arg_type", metadata !"float*", metadata !"float*", metadata !"float*", metadata !"float*", metadata !"uchar", metadata !"uchar", metadata !"uchar", metadata !"uchar"}
!17 = metadata !{metadata !"kernel_arg_type_qual", metadata !"", metadata !"", metadata !"const", metadata !"const", metadata !"const", metadata !"const", metadata !"const", metadata !"const"}
!18 = metadata !{metadata !"kernel_arg_name", metadata !"in", metadata !"out", metadata !"weight", metadata !"biases", metadata !"in_width", metadata !"in_height", metadata !"mask_depth", metadata !"mask_size"}
!19 = metadata !{metadata !"reqd_work_group_size", i32 4, i32 4, i32 2}
!20 = metadata !{void (float addrspace(1)*, float addrspace(1)*, float addrspace(2)*, float addrspace(2)*, i8, i8, i8)* @conv_local_flatmem, metadata !8, metadata !9, metadata !10, metadata !11, metadata !12, metadata !19}
!21 = metadata !{void (float addrspace(1)*, float addrspace(1)*, float addrspace(2)*, float addrspace(2)*)* @fully_connected_local, metadata !22, metadata !23, metadata !24, metadata !25, metadata !26, metadata !27}
!22 = metadata !{metadata !"kernel_arg_addr_space", i32 1, i32 1, i32 2, i32 2}
!23 = metadata !{metadata !"kernel_arg_access_qual", metadata !"none", metadata !"none", metadata !"none", metadata !"none"}
!24 = metadata !{metadata !"kernel_arg_type", metadata !"float*", metadata !"float*", metadata !"float*", metadata !"float*"}
!25 = metadata !{metadata !"kernel_arg_type_qual", metadata !"", metadata !"", metadata !"const", metadata !"const"}
!26 = metadata !{metadata !"kernel_arg_name", metadata !"in", metadata !"out", metadata !"weights", metadata !"biases"}
!27 = metadata !{metadata !"reqd_work_group_size", i32 256, i32 1, i32 1}
!28 = metadata !{void (float addrspace(1)*, float addrspace(1)*)* @softmax_layer, metadata !1, metadata !2, metadata !3, metadata !4, metadata !5, metadata !29}
!29 = metadata !{metadata !"reqd_work_group_size", i32 1, i32 1, i32 1}
!30 = metadata !{float 2.500000e+00}

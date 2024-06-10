using Metal
using Underscores
using Revise
using BenchmarkTools
using ChangePrecision

# Problem 1
function add_ten(a)
    i = thread_position_in_grid_1d()
    a[i] = a[i] + 10
    return
end

a = MtlArray(1:1000*10000)

@metal threads=1000 groups=10000 add_ten(a)

# Problem 2
function gpu_zip(a,b,out)
    i = thread_position_in_grid_1d()
    out[i] = a[i] + b[i]
    return
end

a = MtlArray(1:1000)
b = MtlArray(1:1000)
c = MtlArray(Float32.(zeros(1000)))

@metal threads=1000 groups=1 gpu_zip(a,b,c)
c

# Problem 3
function guarded_gpu_map(a,out,size)
    i = thread_position_in_grid_1d()
    if i <= size
        @inbounds out[i] = a[i] + 10
    end 
    return
end

a = MtlArray(1:100)
out = MtlArray(Float32.(zeros(100)))

@metal threads=1000 groups=1 guarded_gpu_map(a,out,100)

# Problem 4
# FIXME: why are the bounds checks causing the second column to get no updates?
function map_2d(out, a, size)
    (i,j) = thread_position_in_grid_2d()
    if i <= size[1] && j <= size[2]
        @inbounds out[i,j] = a[i,j] + 10
    end
    return
end

out = Float32.(zeros(2,2)) |> MtlArray
a = Float32.(Array(1:2*2)) |> (@_ reshape(__, (2,2))) |> MtlArray

@metal threads=1000 groups=1 map_2d(out,a,(2,2))
out

# Problem 5: Broadcast
rs(shape...) = (@_ reshape(__, shape))
function gpu_bcast(out, a, b, size)
    (i,j) = thread_position_in_grid_2d()
    if i <= size[1] && j <= size[2]
       @inbounds out[i,j] = a[i] + b[j]
    end
    return
end

a = Float32.(Array(0:1)) |> rs(2,1) |> MtlArray
b = Float32.(Array(0:1)) |> rs(1,2) |> MtlArray
out = zeros(Float32, 2, 2) |> MtlArray
@metal threads=(2,2) groups=1 gpu_bcast(out, a, b, (2,2))
out

# Problem 6: Blocks 

# Implement a kernel that adds 10 to each position of a and stores it in out.
# You have fewer threads per block than the size of a.

function add_ten_few_threads(out, a, size)
    # Technically we don't need to do this positioning arithmetic in Metal, but
    # I figured I ought to in the spirit of the exercise.    
    group_idx = threadgroup_position_in_grid_1d()
    group_sz = threads_per_threadgroup_1d()
    thread_idx = thread_index_in_threadgroup()

    i = (group_idx - 1) * group_sz + thread_idx # -1 due to 1-indexing
    if i <= size
        @inbounds out[i] = a[i] + 10
    end
    return
end

out = zeros(Float32, 9) |> MtlArray
a = MtlArray(1:9)
@metal threads = 4 groups = 3 add_ten_few_threads(out, a, 9)
out
Array(out)[9]

# Problem 7
# Blocks 2D: The same kernel as above, but in 2D.
# Fewer theads per block than the size of a.

function map_2d_few_threads(out, a, size)
    # Technically we don't need to do this positioning arithmetic in Metal, but
    # I figured I ought to in the spirit of the exercise.
    group_idx = threadgroup_position_in_grid_1d()
    group_sz = threads_per_threadgroup_1d()
    thread_idx = thread_index_in_threadgroup()
    lin = (group_idx - 1) * group_sz + thread_idx # -1 due to 1-indexing

    i = (lin - 1) ÷ size + 1
    j = (lin - 1) % size + 1

    if i <= size && j <= size
        @inbounds out[i,j] = a[i,j] + 10
    end

    return
end

out = zeros(Float32, (5,5)) |> MtlArray
a = -1 * Float32.(ones(5,5)) |> MtlArray
# Note: you need 7 groups because the stride is only 1. 
# FIXME: Can I change the stride? CUDA seems to have different behavior here
# when you provide a 2D group specification.
@metal threads = (3,3) groups = 7 map_2d_few_threads(out, a, 5)
out

# Problem 8 
# I found the problem description for this one a little unclear so I just did
# something that seemed reasonable while at least using threadgroup-local
# memory.

function map_with_shared_mem(out, a, size)
    group_idx = threadgroup_position_in_grid_1d()
    group_sz = threads_per_threadgroup_1d()
    thread_idx = thread_index_in_threadgroup()
    # FIXME: group array allocation need to be constant sized, add a PR to
    # Metal.jl for better warnings here.
    group_mem = MtlThreadGroupArray(Float32, 3) 

    i = (group_idx - 1) * group_sz + thread_idx # -1 due to 1-indexing

    if i <= size
        group_mem[thread_idx] = a[i]
        threadgroup_barrier(Metal.MemoryFlagThreadGroup) # fence writes to thread group memory

        group_mem[thread_idx] = group_mem[thread_idx] + 10
        threadgroup_barrier(Metal.MemoryFlagThreadGroup)
        out[i] = group_mem[thread_idx]
    end
    return
end

out = zeros(Float32, 30) |> MtlArray
a = -1 * Float32.(ones(30)) |> MtlArray
@metal threads = 3 groups = 10 map_with_shared_mem(out, a, 30)
out

# Problem 10: Dot product
function dot_prod(out, a, b, size)
    group_idx = threadgroup_position_in_grid_1d()
    group_sz = threads_per_threadgroup_1d()
    thread_idx = thread_index_in_threadgroup()
    i = (group_idx - 1) * group_sz + thread_idx # -1 due to 1-indexing
    shared_mem = MtlThreadGroupArray(Float32, 1024) # hardcoded for this problem

    if i <= size
        # load global memory for a into shared memory
        @inbounds my_a = a[i]
        @inbounds my_b = b[i]
        my_prod = my_a * my_b
        shared_mem[i] = my_prod
        threadgroup_barrier(Metal.MemoryFlagThreadGroup) # fence writes to thread group memory

        # This is insanely slow, but we'll fix it in a later exercise.
        s = 0
        for j ∈ 1:1024
            s += shared_mem[j]
        end
        out[1] = s
    end
    return
end

out = zeros(Float32, 1) |> MtlArray
a = Float32.(Array(1:1024)) |> MtlArray

@benchmark Metal.@sync @metal threads = 1024 dot_prod(out, a, a, 1024)
out


# Problem 11: 1D convolution

# Implement a kernel that computes a 1D convolution between arr and filter and
# stores it in out. You need to handle the general case. You only need 2 global
# reads and 1 global write per thread.
function dim1_conv(out, a, b, a_size, b_size)
    group_idx = threadgroup_position_in_grid_1d()
    group_sz = threads_per_threadgroup_1d()
    thread_idx = thread_index_in_threadgroup()
    i = (group_idx - 1) * group_sz + thread_idx # -1 due to 1-indexing
    shared_a = MtlThreadGroupArray(Float32, 1024) # max conv size of 1024
    shared_out = MtlThreadGroupArray(Float32, 1024) # max conv size of 1024
    shared_b = MtlThreadGroupArray(Float32, 1024) # max conv size of 1024


    if i <= b_size
        @inbounds shared_b[i] = b[i]
    end

    if i <= a_size
        @inbounds my_a = a[i]
        @inbounds shared_a[i] = my_a
        @inbounds shared_out[i] = 0
        threadgroup_barrier(Metal.MemoryFlagThreadGroup) # fence writes to thread group memory
        # each a_i computes its own part of the convolution
        for offset ∈ 0:b_size-1
            if i + offset <= a_size
               @inbounds shared_out[i] += shared_a[i+offset] * shared_b[1+offset]
            end
        end
        threadgroup_barrier(Metal.MemoryFlagThreadGroup) # fence writes to thread group memory
        @inbounds out[i] = shared_out[i]
    end
    return
end

out = zeros(Float32, 6) |> MtlArray
a = Float32.(Array(0:5)) |> MtlArray
b = Float32.(Array(0:2)) |> MtlArray

@metal threads = 8 dim1_conv(out, a, b, 6, 3)
out


# Problem 12: Prefix sum

# Implement a kernel that computes a sum over a and stores it in out. If the
# size of a exceeds the threadgroup size, only store the sum of each block.
function ppfx(out, a, size, expilszm1, exp_stride_schedule)
    group_idx = threadgroup_position_in_grid_1d()
    group_sz = threads_per_threadgroup_1d()
    thread_idx = thread_index_in_threadgroup()
    i = (group_idx - 1) * group_sz + thread_idx # -1 due to 1-indexing
    shared_out = MtlThreadGroupArray(Float32, 1024) # max conv size of 1024

    shared_out[thread_idx] = a[i]

    for stride in 1:expilszm1
        threadgroup_barrier(Metal.MemoryFlagThreadGroup) # fence writes to thread group memory    
        # only some threads participate in the sum at each wavefront
        if thread_idx <= size && (thread_idx % exp_stride_schedule[stride+1]) == 0
            @inbounds s = shared_out[thread_idx]
            @inbounds offset = exp_stride_schedule[stride]
            if thread_idx-offset <= size && (thread_idx-1 >= 1)
                @inbounds s += shared_out[thread_idx-offset]    
            end 
            @inbounds shared_out[thread_idx] = s
        end
    end

    threadgroup_barrier(Metal.MemoryFlagThreadGroup) # fence writes to thread group memory    
    
    out[group_idx] = shared_out[size]
    return
end


a = Float32.(Array(1:1024*1000)) |> MtlArray
sz = min(1024, length(a))
groups = Int(ceil(length(a) / 1024))
out = zeros(Float32, groups) |> MtlArray
expilszm1 = Int((2^(log(2,sz)-1)))
exp_stride_schedule = (2 .^ Array(0:expilszm1)) |> MtlArray
@metal threads = sz groups = groups ppfx(out, a, sz, expilszm1, exp_stride_schedule)
out

@assert sum(a) == sum(out)
@benchmark Metal.@sync @metal threads = size groups = groups ppfx(out, a, size, expilszm1, exp_stride_schedule)
a = Array(a)
@benchmark sum(a)

# Problem 13: Axis sum
"A kernel that computes a sum over each column of `a` and stores it in `out`."
function col_sum(out, a, size, expilszm1, exp_stride_schedule)
    col = threadgroup_position_in_grid_1d()
    row = thread_position_in_threadgroup_1d()
    shared_col = MtlThreadGroupArray(Float32, 1024) # must be of fixed size, so hardcoded for now to 1024.
    shared_col[row] = a[row,col]

    for stride in 1:expilszm1
        threadgroup_barrier(Metal.MemoryFlagThreadGroup) # fence writes to thread group memory    
        # only some threads participate in the sum at each wavefront
        if row <= size && (row % exp_stride_schedule[stride+1]) == 0
            @inbounds s = shared_col[row]
            @inbounds offset = exp_stride_schedule[stride]
            if row-offset <= size && (row-1 >= 1)
                @inbounds s += shared_col[row-offset]    
            end 
            @inbounds shared_col[row] = s
        end
    end

    threadgroup_barrier(Metal.MemoryFlagThreadGroup) # fence writes to thread group memory    
    
    out[col] = shared_col[size]
    return
    
end

# TODO: Metal.jl linter:
# - Make sure Float32 everywhere
# - Make sure MtlThreadGroupArray(_, sz) has sz constant.

a = rand(Float32, 1024,10000) |> MtlArray
sz = size(a)[1]
out = zeros(Float32,size(a)[2]) |> MtlArray
expilszm1 = Int(floor(2.0^(log(2,sz)-1)))
exp_stride_schedule = (2 .^ Array(0:expilszm1)) |> MtlArray
@metal threads = sz groups = size(out)[1] col_sum(out, a, sz, expilszm1, exp_stride_schedule)
out
@assert all([sum(Array(a)[:,j]) == Array(out)[j] for j ∈ size(a)[2]])


# Problem 14: Matmul
# Implement a kernel that multiplies square matrices a and b and stores the result in out.

# Tip: The most efficient algorithm here will copy a block into shared memory
# before computing each of the individual row-column dot products. This is easy
# to do if the matrix fits in shared memory. Do that case first. Then update
# your code to compute a partial dot-product and iteratively move the part you
# copied into shared memory. You should be able to do the hard case in 6 global
# reads per thread.
"Multiples `n`×`n` square matrices `a` by `b`, storing the result in `out`."
function matmul(out, a, b, n)
        
end
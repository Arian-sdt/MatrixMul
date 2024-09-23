.section .note.GNU-stack,""
.global assembly_matrix_mul, assembly_matrix_simd_mul
.data
float_one:
    temparr:
    .fill 8, 4, 0

.text

assembly_matrix_mul:
    mov %rcx, %r11           # size

    mov $0, %r8           # i = 0
assembly_matrix_mul_outer_loop:
    cmp %r11, %r8
    je assembly_matrix_mul_outer_loop_ret

    mov $0, %r9           # j = 0
assembly_matrix_mul_middle_loop:
    cmp %r11, %r9
    je assembly_matrix_mul_middle_loop_ret

    mov $0, %r10           # k = 0
assembly_matrix_mul_inner_loop:
    cmp %r11, %r10
    je assembly_matrix_mul_inner_loop_ret

    # Calculate addresses for firstMatrix[i * size + k] and get the value at that point 
    mov %r8, %rcx
    imul %r11, %rcx
    add %r10, %rcx
    movss (%rdi, %rcx, 4), %xmm0 

    # Calculate addresses for secondMatrix[k * size + j] and get the value at that point
    mov %r10, %rcx
    imul %r11, %rcx
    add %r9, %rcx
    movss (%rsi, %rcx, 4), %xmm1 

    mulss %xmm1, %xmm0         # Multiply

    # Calculate addresses for result[i * size + j] and put the calculated value there
    mov %r8, %rcx
    imul %r11, %rcx
    add %r9, %rcx

    movss (%rdx, %rcx, 4), %xmm2    #
    addss %xmm0, %xmm2              # Add result
    movss %xmm2, (%rdx, %rcx, 4)    # Store updated result

    inc %r10
    jmp assembly_matrix_mul_inner_loop

assembly_matrix_mul_inner_loop_ret:
    inc %r9
    jmp assembly_matrix_mul_middle_loop

assembly_matrix_mul_middle_loop_ret:
    inc %r8
    jmp assembly_matrix_mul_outer_loop

assembly_matrix_mul_outer_loop_ret:
    ret





assembly_matrix_simd_mul:
    push %r12
    push %r13 
    push %r14
    push %r15

    lea temparr(%rip), %r14

    mov $8, %r13

    vxorps %ymm8, %ymm8, %ymm8  # calculation register
    vxorps %ymm7, %ymm7, %ymm7  # calculation register

    mov %rcx, %r11           # size

    mov $0, %r8           # i = 0
assembly_matrix_simd_mul_outer_loop:
    cmp %r11, %r8
    je assembly_matrix_simd_mul_outer_loop_ret

    mov $0, %r9           # j = 0
assembly_matrix_simd_mul_middle_loop:
    cmp %r11, %r9
    je assembly_matrix_simd_mul_middle_loop_ret
    
    vxorps %ymm6, %ymm6, %ymm6  # calculation register

    mov $0, %r10           # k = 0
assembly_matrix_simd_mul_inner_loop:
    cmp %r11, %r10
    je assembly_matrix_simd_mul_inner_loop_ret

    # Calculate addresses for firstMatrix[i * size + k] and get the value at that point 

    mov %r8, %rcx
    imul %r11, %rcx
    add %r10, %rcx
    imul $4, %rcx
    add %rdi, %rcx
    vmovups (%rcx), %ymm8 

    mov $0, %r12        # x = 0

assembly_matrix_simd_mul_temparr_load_loop:
    cmp %r12, %r13
    je assembly_matrix_simd_mul_temparr_load_loop_end

    # Calculate addresses for secondMatrix[(k + x) * size + j] and get the value at that point
    mov %r10, %rcx
    add %r12, %rcx 
    imul %r11, %rcx
    add %r9, %rcx
    movd (%rsi, %rcx, 4), %xmm0
    movd %xmm0, (%r14, %r12, 4)

    inc %r12 
    jmp assembly_matrix_simd_mul_temparr_load_loop


assembly_matrix_simd_mul_temparr_load_loop_end:

    vmovups temparr(%rip), %ymm7

    vmulps %ymm7, %ymm8, %ymm8 
    vaddps %ymm8, %ymm6, %ymm6

    add $8, %r10
    jmp assembly_matrix_simd_mul_inner_loop

assembly_matrix_simd_mul_inner_loop_ret:


    pxor %xmm0, %xmm0
    pxor %xmm1, %xmm1
    vextractf128 $1, %ymm6, %xmm1
    vaddps %xmm1, %xmm6, %xmm6
    vshufps $0b00001110, %xmm6, %xmm6, %xmm1
    vaddps %xmm1, %xmm6, %xmm6
    vshufps $0b00000001, %xmm6, %xmm6, %xmm1
    vaddss %xmm1, %xmm6, %xmm0


    # Calculate addresses for result[i * size + j] and put the calculated value there
    mov %r8, %rcx
    imul %r11, %rcx
    add %r9, %rcx

    movss %xmm0, (%rdx, %rcx, 4)    # Store updated result

    inc %r9
    jmp assembly_matrix_simd_mul_middle_loop

assembly_matrix_simd_mul_middle_loop_ret:
    inc %r8
    jmp assembly_matrix_simd_mul_outer_loop

assembly_matrix_simd_mul_outer_loop_ret:

    pop %r15
    pop %r14
    pop %r13 
    pop %r12

    ret


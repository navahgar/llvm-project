; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc < %s -mtriple=i686-- -mattr=+avx | FileCheck %s --check-prefixes=CHECK,X86
; RUN: llc < %s -mtriple=x86_64-- -mattr=+avx2 | FileCheck %s --check-prefixes=CHECK,X64

; PR 9267

define <4 x i16> @func_16_32(ptr %a, ptr %b, ptr %c) nounwind {
; X86-LABEL: func_16_32:
; X86:       # %bb.0:
; X86-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X86-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; X86-NEXT:    movl {{[0-9]+}}(%esp), %edx
; X86-NEXT:    vmovdqa (%edx), %xmm0
; X86-NEXT:    vpaddw (%ecx), %xmm0, %xmm0
; X86-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,4,5,8,9,12,13,8,9,12,13,12,13,14,15]
; X86-NEXT:    vmovq %xmm0, (%eax)
; X86-NEXT:    retl
;
; X64-LABEL: func_16_32:
; X64:       # %bb.0:
; X64-NEXT:    vmovdqa (%rsi), %xmm0
; X64-NEXT:    vpaddw (%rdi), %xmm0, %xmm0
; X64-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,4,5,8,9,12,13,8,9,12,13,12,13,14,15]
; X64-NEXT:    vmovq %xmm0, (%rdx)
; X64-NEXT:    retq
  %F = load <4 x i32>, ptr %a
  %G = trunc <4 x i32> %F to <4 x i16>
  %H = load <4 x i32>, ptr %b
  %Y = trunc <4 x i32> %H to <4 x i16>
  %T = add <4 x i16> %Y, %G
  store <4 x i16>%T , ptr %c
  ret <4 x i16> %T
}

define <4 x i16> @func_16_64(ptr %a, ptr %b, ptr %c) nounwind {
; X86-LABEL: func_16_64:
; X86:       # %bb.0:
; X86-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X86-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; X86-NEXT:    movl {{[0-9]+}}(%esp), %edx
; X86-NEXT:    vmovaps (%edx), %ymm0
; X86-NEXT:    vxorps (%ecx), %ymm0, %ymm0
; X86-NEXT:    vextractf128 $1, %ymm0, %xmm1
; X86-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; X86-NEXT:    vpblendw {{.*#+}} xmm1 = xmm1[0],xmm2[1,2,3],xmm1[4],xmm2[5,6,7]
; X86-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0],xmm2[1,2,3],xmm0[4],xmm2[5,6,7]
; X86-NEXT:    vpackusdw %xmm1, %xmm0, %xmm0
; X86-NEXT:    vpackusdw %xmm0, %xmm0, %xmm0
; X86-NEXT:    vmovq %xmm0, (%eax)
; X86-NEXT:    vzeroupper
; X86-NEXT:    retl
;
; X64-LABEL: func_16_64:
; X64:       # %bb.0:
; X64-NEXT:    vmovdqa (%rsi), %ymm0
; X64-NEXT:    vpxor (%rdi), %ymm0, %ymm0
; X64-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; X64-NEXT:    vpblendw {{.*#+}} ymm0 = ymm0[0],ymm1[1,2,3],ymm0[4],ymm1[5,6,7],ymm0[8],ymm1[9,10,11],ymm0[12],ymm1[13,14,15]
; X64-NEXT:    vextracti128 $1, %ymm0, %xmm1
; X64-NEXT:    vpackusdw %xmm1, %xmm0, %xmm0
; X64-NEXT:    vpackusdw %xmm0, %xmm0, %xmm0
; X64-NEXT:    vmovq %xmm0, (%rdx)
; X64-NEXT:    vzeroupper
; X64-NEXT:    retq
  %F = load <4 x i64>, ptr %a
  %G = trunc <4 x i64> %F to <4 x i16>
  %H = load <4 x i64>, ptr %b
  %Y = trunc <4 x i64> %H to <4 x i16>
  %T = xor <4 x i16> %Y, %G
  store <4 x i16>%T , ptr %c
  ret <4 x i16> %T
}

define <4 x i32> @func_32_64(ptr %a, ptr %b) nounwind {
; X86-LABEL: func_32_64:
; X86:       # %bb.0:
; X86-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X86-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; X86-NEXT:    vmovaps (%ecx), %ymm0
; X86-NEXT:    vorps (%eax), %ymm0, %ymm0
; X86-NEXT:    vextractf128 $1, %ymm0, %xmm1
; X86-NEXT:    vshufps {{.*#+}} xmm0 = xmm0[0,2],xmm1[0,2]
; X86-NEXT:    vzeroupper
; X86-NEXT:    retl
;
; X64-LABEL: func_32_64:
; X64:       # %bb.0:
; X64-NEXT:    vmovaps (%rsi), %ymm0
; X64-NEXT:    vorps (%rdi), %ymm0, %ymm0
; X64-NEXT:    vextractf128 $1, %ymm0, %xmm1
; X64-NEXT:    vshufps {{.*#+}} xmm0 = xmm0[0,2],xmm1[0,2]
; X64-NEXT:    vzeroupper
; X64-NEXT:    retq
  %F = load <4 x i64>, ptr %a
  %G = trunc <4 x i64> %F to <4 x i32>
  %H = load <4 x i64>, ptr %b
  %Y = trunc <4 x i64> %H to <4 x i32>
  %T = or <4 x i32> %Y, %G
  ret <4 x i32> %T
}

define <4 x i8> @func_8_16(ptr %a, ptr %b) nounwind {
; X86-LABEL: func_8_16:
; X86:       # %bb.0:
; X86-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X86-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; X86-NEXT:    vmovq {{.*#+}} xmm0 = mem[0],zero
; X86-NEXT:    vmovq {{.*#+}} xmm1 = mem[0],zero
; X86-NEXT:    vpaddb %xmm0, %xmm1, %xmm0
; X86-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,2,4,6,u,u,u,u,u,u,u,u,u,u,u,u]
; X86-NEXT:    retl
;
; X64-LABEL: func_8_16:
; X64:       # %bb.0:
; X64-NEXT:    movq (%rdi), %rax
; X64-NEXT:    vmovd %eax, %xmm0
; X64-NEXT:    movl %eax, %ecx
; X64-NEXT:    shrl $16, %ecx
; X64-NEXT:    vpinsrb $1, %ecx, %xmm0, %xmm0
; X64-NEXT:    movq %rax, %rcx
; X64-NEXT:    shrq $32, %rcx
; X64-NEXT:    vpinsrb $2, %ecx, %xmm0, %xmm0
; X64-NEXT:    shrq $48, %rax
; X64-NEXT:    vpinsrb $3, %eax, %xmm0, %xmm0
; X64-NEXT:    movq (%rsi), %rax
; X64-NEXT:    vmovd %eax, %xmm1
; X64-NEXT:    movl %eax, %ecx
; X64-NEXT:    shrl $16, %ecx
; X64-NEXT:    vpinsrb $1, %ecx, %xmm1, %xmm1
; X64-NEXT:    movq %rax, %rcx
; X64-NEXT:    shrq $32, %rcx
; X64-NEXT:    vpinsrb $2, %ecx, %xmm1, %xmm1
; X64-NEXT:    shrq $48, %rax
; X64-NEXT:    vpinsrb $3, %eax, %xmm1, %xmm1
; X64-NEXT:    vpaddb %xmm0, %xmm1, %xmm0
; X64-NEXT:    retq
  %F = load <4 x i16>, ptr %a
  %G = trunc <4 x i16> %F to <4 x i8>
  %H = load <4 x i16>, ptr %b
  %Y = trunc <4 x i16> %H to <4 x i8>
  %T = add <4 x i8> %Y, %G
  ret <4 x i8> %T
}

define <4 x i8> @func_8_32(ptr %a, ptr %b) nounwind {
; X86-LABEL: func_8_32:
; X86:       # %bb.0:
; X86-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X86-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; X86-NEXT:    vmovdqa (%ecx), %xmm0
; X86-NEXT:    vpsubb (%eax), %xmm0, %xmm0
; X86-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,4,8,12,u,u,u,u,u,u,u,u,u,u,u,u]
; X86-NEXT:    retl
;
; X64-LABEL: func_8_32:
; X64:       # %bb.0:
; X64-NEXT:    vmovdqa (%rsi), %xmm0
; X64-NEXT:    vpsubb (%rdi), %xmm0, %xmm0
; X64-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,4,8,12,u,u,u,u,u,u,u,u,u,u,u,u]
; X64-NEXT:    retq
  %F = load <4 x i32>, ptr %a
  %G = trunc <4 x i32> %F to <4 x i8>
  %H = load <4 x i32>, ptr %b
  %Y = trunc <4 x i32> %H to <4 x i8>
  %T = sub <4 x i8> %Y, %G
  ret <4 x i8> %T
}

define <4 x i8> @func_8_64(ptr %a, ptr %b) nounwind {
; X86-LABEL: func_8_64:
; X86:       # %bb.0:
; X86-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X86-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; X86-NEXT:    vmovdqa (%ecx), %xmm0
; X86-NEXT:    vmovdqa 16(%ecx), %xmm1
; X86-NEXT:    vmovdqa {{.*#+}} xmm2 = <0,8,u,u,u,u,u,u,u,u,u,u,u,u,u,u>
; X86-NEXT:    vpshufb %xmm2, %xmm1, %xmm1
; X86-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; X86-NEXT:    vpunpcklwd {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; X86-NEXT:    vmovdqa (%eax), %xmm1
; X86-NEXT:    vmovdqa 16(%eax), %xmm3
; X86-NEXT:    vpshufb %xmm2, %xmm3, %xmm3
; X86-NEXT:    vpshufb %xmm2, %xmm1, %xmm1
; X86-NEXT:    vpunpcklwd {{.*#+}} xmm1 = xmm1[0],xmm3[0],xmm1[1],xmm3[1],xmm1[2],xmm3[2],xmm1[3],xmm3[3]
; X86-NEXT:    vpaddb %xmm0, %xmm1, %xmm0
; X86-NEXT:    retl
;
; X64-LABEL: func_8_64:
; X64:       # %bb.0:
; X64-NEXT:    vmovdqa (%rdi), %xmm0
; X64-NEXT:    vmovdqa 16(%rdi), %xmm1
; X64-NEXT:    vpbroadcastw {{.*#+}} xmm2 = [0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8]
; X64-NEXT:    vpshufb %xmm2, %xmm1, %xmm1
; X64-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; X64-NEXT:    vpunpcklwd {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; X64-NEXT:    vmovdqa (%rsi), %xmm1
; X64-NEXT:    vmovdqa 16(%rsi), %xmm3
; X64-NEXT:    vpshufb %xmm2, %xmm3, %xmm3
; X64-NEXT:    vpshufb %xmm2, %xmm1, %xmm1
; X64-NEXT:    vpunpcklwd {{.*#+}} xmm1 = xmm1[0],xmm3[0],xmm1[1],xmm3[1],xmm1[2],xmm3[2],xmm1[3],xmm3[3]
; X64-NEXT:    vpaddb %xmm0, %xmm1, %xmm0
; X64-NEXT:    retq
  %F = load <4 x i64>, ptr %a
  %G = trunc <4 x i64> %F to <4 x i8>
  %H = load <4 x i64>, ptr %b
  %Y = trunc <4 x i64> %H to <4 x i8>
  %T = add <4 x i8> %Y, %G
  ret <4 x i8> %T
}

define <4 x i16> @const_16_32() nounwind {
; CHECK-LABEL: const_16_32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vmovddup {{.*#+}} xmm0 = [0,3,8,7,0,3,8,7]
; CHECK-NEXT:    # xmm0 = mem[0,0]
; CHECK-NEXT:    ret{{[l|q]}}
  %G = trunc <4 x i32> <i32 0, i32 3, i32 8, i32 7> to <4 x i16>
  ret <4 x i16> %G
}

define <4 x i16> @const_16_64() nounwind {
; CHECK-LABEL: const_16_64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vmovddup {{.*#+}} xmm0 = [0,3,8,7,0,3,8,7]
; CHECK-NEXT:    # xmm0 = mem[0,0]
; CHECK-NEXT:    ret{{[l|q]}}
  %G = trunc <4 x i64> <i64 0, i64 3, i64 8, i64 7> to <4 x i16>
  ret <4 x i16> %G
}

define void @bugOnTruncBitwidthReduce() nounwind {
; CHECK-LABEL: bugOnTruncBitwidthReduce:
; CHECK:       # %bb.0: # %meh
; CHECK-NEXT:    ret{{[l|q]}}
meh:
  %0 = xor <4 x i64> zeroinitializer, zeroinitializer
  %1 = trunc <4 x i64> %0 to <4 x i32>
  %2 = lshr <4 x i32> %1, <i32 18, i32 18, i32 18, i32 18>
  %3 = xor <4 x i32> %2, %1
  ret void
}

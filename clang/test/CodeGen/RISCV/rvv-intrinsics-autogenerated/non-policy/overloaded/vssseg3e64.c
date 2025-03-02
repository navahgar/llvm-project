// NOTE: Assertions have been autogenerated by utils/update_cc_test_checks.py UTC_ARGS: --version 2
// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-feature +v -target-feature +zfh \
// RUN:   -target-feature +experimental-zvfh -disable-O0-optnone  \
// RUN:   -emit-llvm %s -o - | opt -S -passes=mem2reg | \
// RUN:   FileCheck --check-prefix=CHECK-RV64 %s

#include <riscv_vector.h>

// CHECK-RV64-LABEL: define dso_local void @test_vssseg3e64_v_f64m1
// CHECK-RV64-SAME: (ptr noundef [[BASE:%.*]], i64 noundef [[BSTRIDE:%.*]], <vscale x 1 x double> [[V0:%.*]], <vscale x 1 x double> [[V1:%.*]], <vscale x 1 x double> [[V2:%.*]], i64 noundef [[VL:%.*]]) #[[ATTR0:[0-9]+]] {
// CHECK-RV64-NEXT:  entry:
// CHECK-RV64-NEXT:    call void @llvm.riscv.vssseg3.nxv1f64.i64(<vscale x 1 x double> [[V0]], <vscale x 1 x double> [[V1]], <vscale x 1 x double> [[V2]], ptr [[BASE]], i64 [[BSTRIDE]], i64 [[VL]])
// CHECK-RV64-NEXT:    ret void
//
void test_vssseg3e64_v_f64m1(double *base, ptrdiff_t bstride, vfloat64m1_t v0, vfloat64m1_t v1, vfloat64m1_t v2, size_t vl) {
  return __riscv_vssseg3e64(base, bstride, v0, v1, v2, vl);
}

// CHECK-RV64-LABEL: define dso_local void @test_vssseg3e64_v_f64m2
// CHECK-RV64-SAME: (ptr noundef [[BASE:%.*]], i64 noundef [[BSTRIDE:%.*]], <vscale x 2 x double> [[V0:%.*]], <vscale x 2 x double> [[V1:%.*]], <vscale x 2 x double> [[V2:%.*]], i64 noundef [[VL:%.*]]) #[[ATTR0]] {
// CHECK-RV64-NEXT:  entry:
// CHECK-RV64-NEXT:    call void @llvm.riscv.vssseg3.nxv2f64.i64(<vscale x 2 x double> [[V0]], <vscale x 2 x double> [[V1]], <vscale x 2 x double> [[V2]], ptr [[BASE]], i64 [[BSTRIDE]], i64 [[VL]])
// CHECK-RV64-NEXT:    ret void
//
void test_vssseg3e64_v_f64m2(double *base, ptrdiff_t bstride, vfloat64m2_t v0, vfloat64m2_t v1, vfloat64m2_t v2, size_t vl) {
  return __riscv_vssseg3e64(base, bstride, v0, v1, v2, vl);
}

// CHECK-RV64-LABEL: define dso_local void @test_vssseg3e64_v_i64m1
// CHECK-RV64-SAME: (ptr noundef [[BASE:%.*]], i64 noundef [[BSTRIDE:%.*]], <vscale x 1 x i64> [[V0:%.*]], <vscale x 1 x i64> [[V1:%.*]], <vscale x 1 x i64> [[V2:%.*]], i64 noundef [[VL:%.*]]) #[[ATTR0]] {
// CHECK-RV64-NEXT:  entry:
// CHECK-RV64-NEXT:    call void @llvm.riscv.vssseg3.nxv1i64.i64(<vscale x 1 x i64> [[V0]], <vscale x 1 x i64> [[V1]], <vscale x 1 x i64> [[V2]], ptr [[BASE]], i64 [[BSTRIDE]], i64 [[VL]])
// CHECK-RV64-NEXT:    ret void
//
void test_vssseg3e64_v_i64m1(int64_t *base, ptrdiff_t bstride, vint64m1_t v0, vint64m1_t v1, vint64m1_t v2, size_t vl) {
  return __riscv_vssseg3e64(base, bstride, v0, v1, v2, vl);
}

// CHECK-RV64-LABEL: define dso_local void @test_vssseg3e64_v_i64m2
// CHECK-RV64-SAME: (ptr noundef [[BASE:%.*]], i64 noundef [[BSTRIDE:%.*]], <vscale x 2 x i64> [[V0:%.*]], <vscale x 2 x i64> [[V1:%.*]], <vscale x 2 x i64> [[V2:%.*]], i64 noundef [[VL:%.*]]) #[[ATTR0]] {
// CHECK-RV64-NEXT:  entry:
// CHECK-RV64-NEXT:    call void @llvm.riscv.vssseg3.nxv2i64.i64(<vscale x 2 x i64> [[V0]], <vscale x 2 x i64> [[V1]], <vscale x 2 x i64> [[V2]], ptr [[BASE]], i64 [[BSTRIDE]], i64 [[VL]])
// CHECK-RV64-NEXT:    ret void
//
void test_vssseg3e64_v_i64m2(int64_t *base, ptrdiff_t bstride, vint64m2_t v0, vint64m2_t v1, vint64m2_t v2, size_t vl) {
  return __riscv_vssseg3e64(base, bstride, v0, v1, v2, vl);
}

// CHECK-RV64-LABEL: define dso_local void @test_vssseg3e64_v_u64m1
// CHECK-RV64-SAME: (ptr noundef [[BASE:%.*]], i64 noundef [[BSTRIDE:%.*]], <vscale x 1 x i64> [[V0:%.*]], <vscale x 1 x i64> [[V1:%.*]], <vscale x 1 x i64> [[V2:%.*]], i64 noundef [[VL:%.*]]) #[[ATTR0]] {
// CHECK-RV64-NEXT:  entry:
// CHECK-RV64-NEXT:    call void @llvm.riscv.vssseg3.nxv1i64.i64(<vscale x 1 x i64> [[V0]], <vscale x 1 x i64> [[V1]], <vscale x 1 x i64> [[V2]], ptr [[BASE]], i64 [[BSTRIDE]], i64 [[VL]])
// CHECK-RV64-NEXT:    ret void
//
void test_vssseg3e64_v_u64m1(uint64_t *base, ptrdiff_t bstride, vuint64m1_t v0, vuint64m1_t v1, vuint64m1_t v2, size_t vl) {
  return __riscv_vssseg3e64(base, bstride, v0, v1, v2, vl);
}

// CHECK-RV64-LABEL: define dso_local void @test_vssseg3e64_v_u64m2
// CHECK-RV64-SAME: (ptr noundef [[BASE:%.*]], i64 noundef [[BSTRIDE:%.*]], <vscale x 2 x i64> [[V0:%.*]], <vscale x 2 x i64> [[V1:%.*]], <vscale x 2 x i64> [[V2:%.*]], i64 noundef [[VL:%.*]]) #[[ATTR0]] {
// CHECK-RV64-NEXT:  entry:
// CHECK-RV64-NEXT:    call void @llvm.riscv.vssseg3.nxv2i64.i64(<vscale x 2 x i64> [[V0]], <vscale x 2 x i64> [[V1]], <vscale x 2 x i64> [[V2]], ptr [[BASE]], i64 [[BSTRIDE]], i64 [[VL]])
// CHECK-RV64-NEXT:    ret void
//
void test_vssseg3e64_v_u64m2(uint64_t *base, ptrdiff_t bstride, vuint64m2_t v0, vuint64m2_t v1, vuint64m2_t v2, size_t vl) {
  return __riscv_vssseg3e64(base, bstride, v0, v1, v2, vl);
}

// CHECK-RV64-LABEL: define dso_local void @test_vssseg3e64_v_f64m1_m
// CHECK-RV64-SAME: (<vscale x 1 x i1> [[MASK:%.*]], ptr noundef [[BASE:%.*]], i64 noundef [[BSTRIDE:%.*]], <vscale x 1 x double> [[V0:%.*]], <vscale x 1 x double> [[V1:%.*]], <vscale x 1 x double> [[V2:%.*]], i64 noundef [[VL:%.*]]) #[[ATTR0]] {
// CHECK-RV64-NEXT:  entry:
// CHECK-RV64-NEXT:    call void @llvm.riscv.vssseg3.mask.nxv1f64.i64(<vscale x 1 x double> [[V0]], <vscale x 1 x double> [[V1]], <vscale x 1 x double> [[V2]], ptr [[BASE]], i64 [[BSTRIDE]], <vscale x 1 x i1> [[MASK]], i64 [[VL]])
// CHECK-RV64-NEXT:    ret void
//
void test_vssseg3e64_v_f64m1_m(vbool64_t mask, double *base, ptrdiff_t bstride, vfloat64m1_t v0, vfloat64m1_t v1, vfloat64m1_t v2, size_t vl) {
  return __riscv_vssseg3e64(mask, base, bstride, v0, v1, v2, vl);
}

// CHECK-RV64-LABEL: define dso_local void @test_vssseg3e64_v_f64m2_m
// CHECK-RV64-SAME: (<vscale x 2 x i1> [[MASK:%.*]], ptr noundef [[BASE:%.*]], i64 noundef [[BSTRIDE:%.*]], <vscale x 2 x double> [[V0:%.*]], <vscale x 2 x double> [[V1:%.*]], <vscale x 2 x double> [[V2:%.*]], i64 noundef [[VL:%.*]]) #[[ATTR0]] {
// CHECK-RV64-NEXT:  entry:
// CHECK-RV64-NEXT:    call void @llvm.riscv.vssseg3.mask.nxv2f64.i64(<vscale x 2 x double> [[V0]], <vscale x 2 x double> [[V1]], <vscale x 2 x double> [[V2]], ptr [[BASE]], i64 [[BSTRIDE]], <vscale x 2 x i1> [[MASK]], i64 [[VL]])
// CHECK-RV64-NEXT:    ret void
//
void test_vssseg3e64_v_f64m2_m(vbool32_t mask, double *base, ptrdiff_t bstride, vfloat64m2_t v0, vfloat64m2_t v1, vfloat64m2_t v2, size_t vl) {
  return __riscv_vssseg3e64(mask, base, bstride, v0, v1, v2, vl);
}

// CHECK-RV64-LABEL: define dso_local void @test_vssseg3e64_v_i64m1_m
// CHECK-RV64-SAME: (<vscale x 1 x i1> [[MASK:%.*]], ptr noundef [[BASE:%.*]], i64 noundef [[BSTRIDE:%.*]], <vscale x 1 x i64> [[V0:%.*]], <vscale x 1 x i64> [[V1:%.*]], <vscale x 1 x i64> [[V2:%.*]], i64 noundef [[VL:%.*]]) #[[ATTR0]] {
// CHECK-RV64-NEXT:  entry:
// CHECK-RV64-NEXT:    call void @llvm.riscv.vssseg3.mask.nxv1i64.i64(<vscale x 1 x i64> [[V0]], <vscale x 1 x i64> [[V1]], <vscale x 1 x i64> [[V2]], ptr [[BASE]], i64 [[BSTRIDE]], <vscale x 1 x i1> [[MASK]], i64 [[VL]])
// CHECK-RV64-NEXT:    ret void
//
void test_vssseg3e64_v_i64m1_m(vbool64_t mask, int64_t *base, ptrdiff_t bstride, vint64m1_t v0, vint64m1_t v1, vint64m1_t v2, size_t vl) {
  return __riscv_vssseg3e64(mask, base, bstride, v0, v1, v2, vl);
}

// CHECK-RV64-LABEL: define dso_local void @test_vssseg3e64_v_i64m2_m
// CHECK-RV64-SAME: (<vscale x 2 x i1> [[MASK:%.*]], ptr noundef [[BASE:%.*]], i64 noundef [[BSTRIDE:%.*]], <vscale x 2 x i64> [[V0:%.*]], <vscale x 2 x i64> [[V1:%.*]], <vscale x 2 x i64> [[V2:%.*]], i64 noundef [[VL:%.*]]) #[[ATTR0]] {
// CHECK-RV64-NEXT:  entry:
// CHECK-RV64-NEXT:    call void @llvm.riscv.vssseg3.mask.nxv2i64.i64(<vscale x 2 x i64> [[V0]], <vscale x 2 x i64> [[V1]], <vscale x 2 x i64> [[V2]], ptr [[BASE]], i64 [[BSTRIDE]], <vscale x 2 x i1> [[MASK]], i64 [[VL]])
// CHECK-RV64-NEXT:    ret void
//
void test_vssseg3e64_v_i64m2_m(vbool32_t mask, int64_t *base, ptrdiff_t bstride, vint64m2_t v0, vint64m2_t v1, vint64m2_t v2, size_t vl) {
  return __riscv_vssseg3e64(mask, base, bstride, v0, v1, v2, vl);
}

// CHECK-RV64-LABEL: define dso_local void @test_vssseg3e64_v_u64m1_m
// CHECK-RV64-SAME: (<vscale x 1 x i1> [[MASK:%.*]], ptr noundef [[BASE:%.*]], i64 noundef [[BSTRIDE:%.*]], <vscale x 1 x i64> [[V0:%.*]], <vscale x 1 x i64> [[V1:%.*]], <vscale x 1 x i64> [[V2:%.*]], i64 noundef [[VL:%.*]]) #[[ATTR0]] {
// CHECK-RV64-NEXT:  entry:
// CHECK-RV64-NEXT:    call void @llvm.riscv.vssseg3.mask.nxv1i64.i64(<vscale x 1 x i64> [[V0]], <vscale x 1 x i64> [[V1]], <vscale x 1 x i64> [[V2]], ptr [[BASE]], i64 [[BSTRIDE]], <vscale x 1 x i1> [[MASK]], i64 [[VL]])
// CHECK-RV64-NEXT:    ret void
//
void test_vssseg3e64_v_u64m1_m(vbool64_t mask, uint64_t *base, ptrdiff_t bstride, vuint64m1_t v0, vuint64m1_t v1, vuint64m1_t v2, size_t vl) {
  return __riscv_vssseg3e64(mask, base, bstride, v0, v1, v2, vl);
}

// CHECK-RV64-LABEL: define dso_local void @test_vssseg3e64_v_u64m2_m
// CHECK-RV64-SAME: (<vscale x 2 x i1> [[MASK:%.*]], ptr noundef [[BASE:%.*]], i64 noundef [[BSTRIDE:%.*]], <vscale x 2 x i64> [[V0:%.*]], <vscale x 2 x i64> [[V1:%.*]], <vscale x 2 x i64> [[V2:%.*]], i64 noundef [[VL:%.*]]) #[[ATTR0]] {
// CHECK-RV64-NEXT:  entry:
// CHECK-RV64-NEXT:    call void @llvm.riscv.vssseg3.mask.nxv2i64.i64(<vscale x 2 x i64> [[V0]], <vscale x 2 x i64> [[V1]], <vscale x 2 x i64> [[V2]], ptr [[BASE]], i64 [[BSTRIDE]], <vscale x 2 x i1> [[MASK]], i64 [[VL]])
// CHECK-RV64-NEXT:    ret void
//
void test_vssseg3e64_v_u64m2_m(vbool32_t mask, uint64_t *base, ptrdiff_t bstride, vuint64m2_t v0, vuint64m2_t v1, vuint64m2_t v2, size_t vl) {
  return __riscv_vssseg3e64(mask, base, bstride, v0, v1, v2, vl);
}


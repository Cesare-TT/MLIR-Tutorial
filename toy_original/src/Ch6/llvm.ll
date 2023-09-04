; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@frmt_spec = internal constant [4 x i8] c"%f \00"

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr nocapture noundef readonly, ...) local_unnamed_addr #0

; Function Attrs: nofree nounwind
define void @main() local_unnamed_addr #0 !dbg !3 {
.preheader3:
  %0 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double 1.000000e+00), !dbg !6
  %1 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double 1.600000e+01), !dbg !6
  %putchar = tail call i32 @putchar(i32 10), !dbg !6
  %2 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double 4.000000e+00), !dbg !6
  %3 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double 2.500000e+01), !dbg !6
  %putchar.1 = tail call i32 @putchar(i32 10), !dbg !6
  %4 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double 9.000000e+00), !dbg !6
  %5 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double 3.600000e+01), !dbg !6
  %putchar.2 = tail call i32 @putchar(i32 10), !dbg !6
  ret void, !dbg !7
}

; Function Attrs: nofree nounwind
declare noundef i32 @putchar(i32 noundef) local_unnamed_addr #0

attributes #0 = { nofree nounwind }

!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_C, file: !2, producer: "MLIR", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!2 = !DIFile(filename: "codegen.toy", directory: "mlir/test/Examples/Toy/Ch6")
!3 = distinct !DISubprogram(name: "main", linkageName: "main", scope: !2, file: !2, line: 8, type: !4, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !1)
!4 = !DISubroutineType(cc: DW_CC_normal, types: !5)
!5 = !{}
!6 = !DILocation(line: 13, column: 3, scope: !3)
!7 = !DILocation(line: 8, column: 1, scope: !3)
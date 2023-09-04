# MLIR Tutorial

## AST

在parser.h中增加对新增BinOp的支持：

```C++
  /// Get the precedence of the pending binary operator token.
  int getTokPrecedence() {
    if (!isascii(lexer.getCurToken()))
      return -1;

    // 1 is lowest precedence.
    switch (static_cast<char>(lexer.getCurToken())) {
    // 在此为新增BinOp设定运算优先级
    case '&':
      return 15;
    case '|':
      return 10;
    case '-':
      return 20;
    case '+':
      return 20;
    case '*':
      return 40;
    default:
      return -1;
    }
  }
```

# MLIR
## MLIRGen

1. 通过ODS框架定义新增的BinOp

```TableGen
    def SubOp : Toy_Op<"sub",
        [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
        let summary = "element-wise substraction operation";
        let description = [{
            The "sub" operation performs element-wise substraction between two tensors.
            The shapes of the tensor operands are expected to match.
        }];

        let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
        let results = (outs F64Tensor);

        // Indicate that the operation has a custom parser and printer method.
        let hasCustomAssemblyFormat = 1;

        // Allow building an SubOp with from the two input operands.
        let builders = [
            OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
        ];
    }
```

2. 在MLIRGen中关联运算符和新增的BinOp

```C++
  mlir::Value mlirGen(BinaryExprAST &binop) {
    mlir::Value lhs = mlirGen(*binop.getLHS());
    if (!lhs)
      return nullptr;
    mlir::Value rhs = mlirGen(*binop.getRHS());
    if (!rhs)
      return nullptr;
    auto location = loc(binop.loc());

    // Derive the operation name from the binary operator. At the moment we only
    // 在此将新增的BinOp与运算符进行关联
    switch (binop.getOp()) {
    case '&':
      return builder.create<AndOp>(location, lhs, rhs);
    case '|':
      return builder.create<OrOp>(location, lhs, rhs);
    case '+':
      return builder.create<AddOp>(location, lhs, rhs);
    case '-':
      return builder.create<SubOp>(location, lhs, rhs);
    case '*':
      return builder.create<MulOp>(location, lhs, rhs);
    }

    emitError(location, "invalid binary operator '") << binop.getOp() << "'";
    return nullptr;
  }
```

3. 在Dialect.cpp中声明新增BinOp类的相关function

```C++
    //===----------------------------------------------------------------------===//
    // SubOp
    //===----------------------------------------------------------------------===//

    void SubOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    mlir::Value lhs, mlir::Value rhs) {
    state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
    state.addOperands({lhs, rhs});
    }

    mlir::ParseResult SubOp::parse(mlir::OpAsmParser &parser,
                                mlir::OperationState &result) {
    return parseBinaryOp(parser, result);
    }

    void SubOp::print(mlir::OpAsmPrinter &p) { printBinaryOp(p, *this); }

    /// Infer the output shape of the SubOp, this is required by the shape inference
    /// interface.
    void SubOp::inferShapes() { getResult().setType(getLhs().getType()); }
```

## MLIR Lowering

1. 将运算新增的BinOp lowering到Affine Dialect
    
    ```C++
    <LoweringToAffineLoops.cpp>
    ...
    using SubOpLowering  = BinaryOpLowering<toy::SubOp, arith::SubFOp>;
    using MulOpLowering  = BinaryOpLowering<toy::MulOp, arith::MulFOp>;
    using AnddOpLowering = BinaryOpLowering<toy::AndOp, arith::AndIOp>;
    using OrOpLowering   = BinaryOpLowering<toy::OrOp,  arith::OrIOp>;
    ...
    void ToyToAffineLoweringPass::runOnOperation() {
      // The first thing to define is the conversion target. This will define the
      // final target for this lowering.
      ConversionTarget target(getContext());

      // We define the specific operations, or dialects, that are legal targets for
      // this lowering. In our case, we are lowering to a combination of the
      // `Affine`, `Arith`, `Func`, and `MemRef` dialects.
      target.addLegalDialect<affine::AffineDialect, BuiltinDialect,
                            arith::ArithDialect, func::FuncDialect,
                            memref::MemRefDialect>();

      // We also define the Toy dialect as Illegal so that the conversion will fail
      // if any of these operations are *not* converted. Given that we actually want
      // a partial lowering, we explicitly mark the Toy operations that don't want
      // to lower, `toy.print`, as `legal`. `toy.print` will still need its operands
      // to be updated though (as we convert from TensorType to MemRefType), so we
      // only treat it as `legal` if its operands are legal.
      target.addIllegalDialect<toy::ToyDialect>();
      target.addDynamicallyLegalOp<toy::PrintOp>([](toy::PrintOp op) {
        return llvm::none_of(op->getOperandTypes(),
                            [](Type type) { return llvm::isa<TensorType>(type); });
      });

      // Now that the conversion target has been defined, we just need to provide
      // the set of patterns that will lower the Toy operations.
      RewritePatternSet patterns(&getContext());
      patterns.add<AddOpLowering, SubOpLowering, MulOpLowering, AnddOpLowering, OrOpLowering, ConstantOpLowering, FuncOpLowering, MulOpLowering,
                  PrintOpLowering, ReturnOpLowering, TransposeOpLowering>(
          &getContext());

      // With the target and rewrite patterns defined, we can now attempt the
      // conversion. The conversion will signal failure if any of our `illegal`
      // operations were not converted successfully.
      if (failed(
              applyPartialConversion(getOperation(), target, std::move(patterns))))
        signalPassFailure();
    }
    ```

2. 新增的BinOp已经lowering到affine，因此在完全lowering到LLVM时无需额外操作

# Run

1. 在codegen.toy中使用新增的BinOp：

```
def main() {
  var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
  var b<2, 3> = [1, 2, 3, 4, 5, 6];
  var aa = 187263287;
  var bb = 123487263;
  var c = multiply_transpose(a, b);
  var d = multiply_transpose(b, a);
  print(d);
  
  # new operations
  var e = aa | bb;
  var f = aa & bb;
  var g = a + b;
  var h = a - b;
  var i = a * b;

  print(e);
  print(f);
  print(g);
  print(h);
  print(i);
}
```

2. 执行以下命令`./build/bin/toyc-ch6 -emit=mlir ./toy_mod/src/Ch6/codegen.toy -opt`可以得到生成MLIR

```MLIR
module {
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    %1 = toy.constant dense<0x41A652D26E000000> : tensor<f64>
    %2 = toy.constant dense<0x419D71107C000000> : tensor<f64>
    %3 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
    %4 = toy.mul %3, %3 : tensor<3x2xf64>
    toy.print %4 : tensor<3x2xf64>
    %5 = toy.or %1, %2 : tensor<f64>
    %6 = toy.and %1, %2 : tensor<f64>
    %7 = toy.add %0, %0 : tensor<2x3xf64>
    %8 = toy.sub %0, %0 : tensor<2x3xf64>
    %9 = toy.mul %0, %0 : tensor<2x3xf64>
    toy.print %5 : tensor<f64>
    toy.print %6 : tensor<f64>
    toy.print %7 : tensor<2x3xf64>
    toy.print %8 : tensor<2x3xf64>
    toy.print %9 : tensor<2x3xf64>
    toy.return
  }
}
```

3. 
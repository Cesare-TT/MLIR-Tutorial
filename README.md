# MLIR Tutorial

本文在Toy Example Ch6的基础上，尝试增加一些基础运算符的支持。

## AST

在parser.h中增加对新增BinOp的支持：

```c++
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

## MLIR

### MLIRGen

1. 通过ODS框架,参照已有的And/Mul Op,定义新增的Sub/And/Or BinOp

    ```TableGen
    //===----------------------------------------------------------------------===//
    // SubOp
    //===----------------------------------------------------------------------===//
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

    //===----------------------------------------------------------------------===//
    // AndOp
    //===----------------------------------------------------------------------===//

    def AndOp : Toy_Op<"and",
        [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
      let summary = "element-wise and operation";
      let description = [{
        The "or" operation performs element-wise logic and between two tensors.
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

    //===----------------------------------------------------------------------===//
    // OrOp
    //===----------------------------------------------------------------------===//

    def OrOp : Toy_Op<"or",
        [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
      let summary = "element-wise or operation";
      let description = [{
        The "or" operation performs element-wise logic or between two tensors.
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

    ```c++
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

    ```c++
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

    //===----------------------------------------------------------------------===//
    // AndOp
    //===----------------------------------------------------------------------===//

    void AndOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    mlir::Value lhs, mlir::Value rhs) {
    state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
    state.addOperands({lhs, rhs});
    }

    mlir::ParseResult AndOp::parse(mlir::OpAsmParser &parser,
                                mlir::OperationState &result) {
    return parseBinaryOp(parser, result);
    }

    void AndOp::print(mlir::OpAsmPrinter &p) { printBinaryOp(p, *this); }

    /// Infer the output shape of the AndOp, this is required by the shape inference
    /// interface.
    void AndOp::inferShapes() { getResult().setType(getLhs().getType()); }

    //===----------------------------------------------------------------------===//
    // OrOp
    //===----------------------------------------------------------------------===//

    void OrOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    mlir::Value lhs, mlir::Value rhs) {
    state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
    state.addOperands({lhs, rhs});
    }

    mlir::ParseResult OrOp::parse(mlir::OpAsmParser &parser,
                                mlir::OperationState &result) {
    return parseBinaryOp(parser, result);
    }

    void OrOp::print(mlir::OpAsmPrinter &p) { printBinaryOp(p, *this); }

    /// Infer the output shape of the OrOp, this is required by the shape inference
    /// interface.
    void OrOp::inferShapes() { getResult().setType(getLhs().getType()); }
    ```

4. 在codegen.toy中使用新增的BinOp

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

    执行以下命令`./build/bin/toyc-ch6 -emit=mlir ./toy_mod/src/Ch6/codegen.toy -opt`可以生成如下MLIR：

    ```mlir
    module {
      toy.func @main() {
        %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
        %1 = toy.constant dense<0x419D71107C000000> : tensor<f64>
        %2 = toy.constant dense<0x41A652D26E000000> : tensor<f64>
        %3 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
        %4 = toy.mul %3, %3 : tensor<3x2xf64>
        toy.print %4 : tensor<3x2xf64>
        %5 = toy.or %2, %1 : tensor<f64>
        %6 = toy.and %2, %1 : tensor<f64>
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

### MLIR Lowering

#### 将新增的BinOp lowering到Affine Dialect

1. 为新增Op定义Lowering

    ```c++
    // <LoweringToAffineLoops.cpp>
    /*...*/
    using SubOpLowering  = BinaryOpLowering<toy::SubOp, arith::SubFOp>;
    using MulOpLowering  = BinaryOpLowering<toy::MulOp, arith::MulFOp>;
    using AnddOpLowering = BinaryOpLowering<toy::AndOp, arith::AndIOp>;
    using OrOpLowering   = BinaryOpLowering<toy::OrOp,  arith::OrIOp>;
    /*...*/
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

    此时执行`./build/bin/toyc-ch6 -emit=mlir-affine ./toy_mod/src/Ch6/codegen.toy -opt`命令生成lowering到affine dialect的mlir时，会报如下错误：

    ```
    loc("./toy_mod/src/Ch6/codegen.toy":18:16): error: 'arith.ori' op operand #0 must be signless-integer-like, but got 'f64'
    ```

    这是因为Toy定义的所有数据类型均为F64，为了实现And/Or操作，我们需要为Toy添加整数类型的支持。

#### 为Toy增加整数类型定义的支持

1. 在Lexer中增加对varint/integer/float的识别

    ```c++
    // <Lexer.h>
    enum Token : int {
      tok_semicolon = ';',
      tok_parenthese_open = '(',
      tok_parenthese_close = ')',
      tok_bracket_open = '{',
      tok_bracket_close = '}',
      tok_sbracket_open = '[',
      tok_sbracket_close = ']',

      tok_eof = -1,

      // commands
      tok_return = -2,
      tok_var = -3,
    // 1. 在enum Token中增加tok_varint
      tok_varint = -4,
      tok_def = -5,

      // primary
      tok_identifier = -6,
      tok_number = -7,
    };

    // 2. 定义float/integer类型判定的枚举
    enum ValueType : bool {
      type_float = 1,
      type_integer = 0,
    };
    ...

    // 3. 将Class Lexer的getValue()函数扩展为getIntValue(), getFloatValue()以及getValueType()
    ValueType getValueType() {
      return valueType;
    }

    int64_t getIntValue() {
      assert(curTok == tok_number);
      // if (valueType == type_integer)
      return intnumVal;
    }

    double getFloatValue() {
      assert(curTok == tok_number);
      // else if (valueType == type_float)
      return floatnumVal;
    }

    ...

    Token getTok() {
      ...
      // Identifier: [a-zA-Z][a-zA-Z0-9_]*
      if (isalpha(lastChar)) {
        identifierStr = (char)lastChar;
        while (isalnum((lastChar = Token(getNextChar()))) || lastChar == '_')
          identifierStr += (char)lastChar;

        if (identifierStr == "return")
          return tok_return;
        if (identifierStr == "def")
          return tok_def;
    // 4. 在getTok()函数中增加对varint标识符的识别
        if (identifierStr == "varint")
          return tok_varint;
        if (identifierStr == "var")
          return tok_var;
        return tok_identifier;
      }

      // Number: [0-9.]+
      // 5. 通过解析是否包含小数点，判断给定的数值为integer还是float
      if (isdigit(lastChar) || lastChar == '.') {
        std::string numStr;
        int numisfloat = 0;
        do {
          numStr += lastChar;
          lastChar = Token(getNextChar());
          if (lastChar == '.')
            numisfloat=1;
        } while (isdigit(lastChar) || lastChar == '.');
        if (numisfloat)
          valueType = type_float;
        else
          valueType = type_integer;
        floatnumVal = strtod(numStr.c_str(), nullptr);
        intnumVal   = strtoll(numStr.c_str(), nullptr, 10);
        return tok_number;
      }

      ...

    }

    ...

    ValueType valueType;
    int64_t   intnumVal = 0;
    double    floatnumVal = 0;
    ```

2. 在AST中增加对varint/integer/float的支持

    ```c++
    // <AST.h>

    class ExprAST {
    public:
      enum ExprASTKind {
        Expr_VarDecl,
        // 1. 增加整数类型声明语法
        Expr_VarIntDecl,
        Expr_Return,
        Expr_Num,
        Expr_Literal,
        Expr_Var,
        Expr_BinOp,
        Expr_Call,
        Expr_Print,
      }

    ...

    }

    // 2. 扩展NumberExprAST使其支持integer和float
    class NumberExprAST : public ExprAST {
      double    floatval;
      int64_t   intval;

    public:
      ValueType valueType;

      NumberExprAST(Location loc, ValueType valueType, int64_t intval, double floatval)
          : ExprAST(Expr_Num, std::move(loc)), valueType(valueType), intval(intval), floatval(floatval) {}


      int64_t getIntValue() {
        // if (valueType == type_integer)
        return intval;
      }

      double getFloatValue() {
        // else if (valueType == type_float)
        return floatval;
      }

      /// LLVM style RTTI
      static bool classof(const ExprAST *c) { return c->getKind() == Expr_Num; }
    };

    // 3. 扩展LiteralExprAST使其能够区分数据类型
    class LiteralExprAST : public ExprAST {
      std::vector<std::unique_ptr<ExprAST>> values;
      std::vector<int64_t> dims;

    public:
      ValueType  valueType;

      LiteralExprAST(Location loc, ValueType valueType, std::vector<std::unique_ptr<ExprAST>> values,
                    std::vector<int64_t> dims)
          : ExprAST(Expr_Literal, std::move(loc)), valueType(valueType), values(std::move(values)),
            dims(std::move(dims)) {}

      llvm::ArrayRef<std::unique_ptr<ExprAST>> getValues() { return values; }
      llvm::ArrayRef<int64_t> getDims() { return dims; }

      /// LLVM style RTTI
      static bool classof(const ExprAST *c) { return c->getKind() == Expr_Literal; }

      ValueType getValueType() {
        return valueType;
      }
    };

    ...

    // 4. 增加VarintExprAST class实现整数类型的变量声明 
    /// Expression class for defining a integer variable.
    class VarIntDeclExprAST : public ExprAST {
      std::string name;
      VarType type;
      std::unique_ptr<ExprAST> initVal;

    public:
      VarIntDeclExprAST(Location loc, llvm::StringRef name, VarType type,
                    std::unique_ptr<ExprAST> initVal)
          : ExprAST(Expr_VarIntDecl, std::move(loc)), name(name),
            type(std::move(type)), initVal(std::move(initVal)) {}

      llvm::StringRef getName() { return name; }
      ExprAST *getInitVal() { return initVal.get(); }
      const VarType &getType() { return type; }

      /// LLVM style RTTI
      static bool classof(const ExprAST *c) { return c->getKind() == Expr_VarIntDecl; }
    };



    // <AST.cpp>
    class ASTDumper {
    public:
      void dump(ModuleAST *node);

    private:
      void dump(const VarType &type);
      void dump(VarDeclExprAST *varDecl);
      // 5. 声明VarIntDeclExprAST的dump函数
      void dump(VarIntDeclExprAST *varintDecl);
      void dump(ExprAST *expr);
      void dump(ExprASTList *exprList);
      void dump(NumberExprAST *num);
      void dump(LiteralExprAST *node);
      void dump(VariableExprAST *node);
      void dump(ReturnExprAST *node);
      void dump(BinaryExprAST *node);
      void dump(CallExprAST *node);
      void dump(PrintExprAST *node);
      void dump(PrototypeAST *node);
      void dump(FunctionAST *node);
    }

    ...

    void ASTDumper::dump(ExprAST *expr) {
      llvm::TypeSwitch<ExprAST *>(expr)
          // 6. 增加对VarIntDeclExprAST的处理
          .Case<BinaryExprAST, CallExprAST, LiteralExprAST, NumberExprAST,
                PrintExprAST, ReturnExprAST, VarDeclExprAST, VarIntDeclExprAST, VariableExprAST>(
              [&](auto *node) { this->dump(node); })
          .Default([&](ExprAST *) {
            // No match, fallback to a generic message
            INDENT();
            llvm::errs() << "<unknown Expr, kind " << expr->getKind() << ">\n";
          });
    }

    // 7. 定义VarIntDeclExprAST的dump函数
    void ASTDumper::dump(VarIntDeclExprAST *varintDecl) {
      INDENT();
      llvm::errs() << "VarIntDecl " << varintDecl->getName();
      dump(varintDecl->getType());
      llvm::errs() << " " << loc(varintDecl) << "\n";
      dump(varintDecl->getInitVal());
    }

    ...

    // 8. 在NumberExprAST的ASTDumper增加对整数类型的支持
    void ASTDumper::dump(NumberExprAST *num) {
      INDENT();
      if (num->valueType == type_float) {
        llvm::errs() << num->getIntValue() << " " << loc(num) << "\n";
      } else if (num->valueType == type_integer) {
        llvm::errs() << num->getFloatValue() << " " << loc(num) << "\n";
      }
    }

    ...

    void printLitHelper(ExprAST *litOrNum) {
      if (auto *num = llvm::dyn_cast<NumberExprAST>(litOrNum)) {
        // 9. 增加对整数类型的支持
        if (num->valueType == type_float) {
          llvm::errs() << num->getIntValue();
        } else if (num->valueType == type_integer) {
          llvm::errs() << num->getFloatValue();
        }
        return;
      }

      ...

    }

    ```

3. 在parser中适配对整数类型的支持

    ```c++
    // <Parser.h>

    class Parser {

      ...

      /// Parse a literal number.
      /// numberexpr ::= number
      std::unique_ptr<ExprAST> parseNumberExpr() {
        auto loc = lexer.getLastLocation();
        auto result =
            std::make_unique<NumberExprAST>(std::move(loc),
            // 1. 适配NumberExprAST区分整数类型和浮点类型后的改动
            lexer.getValueType(), lexer.getIntValue(), lexer.getFloatValue());
        lexer.consume(tok_number);
        return std::move(result);
      }

      std::unique_ptr<ExprAST> parseTensorLiteralExpr() {

        ...

        return std::make_unique<LiteralExprAST>(std::move(loc),
          // 2. 适配LiteralExprAST区分整数类型和浮点类型后的改动
          lexer.getValueType(), std::move(values), std::move(dims));
      }

      ...

        /// type ::= < shape_list >
        /// shape_list ::= num | num , shape_list
        std::unique_ptr<VarType> parseType() {

          ...

          while (lexer.getCurToken() == tok_number) {
            // 3. 适配Lexer区分整数类型和浮点类型后的改动
            type->shape.push_back(lexer.getIntValue());
            lexer.getNextToken();
            if (lexer.getCurToken() == ',')
              lexer.getNextToken();
          }

          ...

        }

        // 4. 增加对整数类型声明的解析函数
        /// Parse a integer variable declaration, it starts with a `varint` keyword followed by
        /// and identifier and an optional type (shape specification) before the
        /// initializer.
        /// decl ::= varint identifier [ type ] = expr
        std::unique_ptr<VarIntDeclExprAST> parseIntDeclaration() {
          if (lexer.getCurToken() != tok_varint)
            return parseError<VarIntDeclExprAST>("varint", "to begin declaration");
          auto loc = lexer.getLastLocation();
          lexer.getNextToken(); // eat varint

          if (lexer.getCurToken() != tok_identifier)
            return parseError<VarIntDeclExprAST>("identified",
                                              "after 'varint' declaration");
          std::string id(lexer.getId());
          lexer.getNextToken(); // eat id

          std::unique_ptr<VarType> type; // Type is optional, it can be inferred
          if (lexer.getCurToken() == '<') {
            type = parseType();
            if (!type)
              return nullptr;
          }

          if (!type)
            type = std::make_unique<VarType>();
          lexer.consume(Token('='));
          auto expr = parseExpression();
          return std::make_unique<VarIntDeclExprAST>(std::move(loc), std::move(id),
                                                  std::move(*type), std::move(expr));
        }

        std::unique_ptr<ExprASTList> parseBlock() {

          ...

          while (lexer.getCurToken() != '}' && lexer.getCurToken() != tok_eof) {
            if (lexer.getCurToken() == tok_var) {
              // Variable declaration
              auto varDecl = parseDeclaration();
              if (!varDecl)
                return nullptr;
              exprList->push_back(std::move(varDecl));
            // 5. 增加对整数类型变量声明的解析判定
            } else if (lexer.getCurToken() == tok_varint) {
              // Integer Variable declaration
              auto varintDecl = parseIntDeclaration();
              if (!varintDecl)
                return nullptr;
              exprList->push_back(std::move(varintDecl));
            } else if (lexer.getCurToken() == tok_return) {
              // Return statement
              auto ret = parseReturn();
              if (!ret)
                return nullptr;
              exprList->push_back(std::move(ret));
            } else {
              // General expression
              auto expr = parseExpression();
              if (!expr)
                return nullptr;
              exprList->push_back(std::move(expr));
            }

            ...

          }
        }
    }
    ```

4. 通过ODS框架修改现有Op使其兼容整数

    ```TableGen

    //===----------------------------------------------------------------------===//
    // ConstantIntOp
    //===----------------------------------------------------------------------===//

    // We define a toy operation by inheriting from our base 'Toy_Op' class above.
    // Here we provide the mnemonic and a list of traits for the operation. The
    // constant operation is marked as 'Pure' as it is a pure operation
    // and may be removed if dead.
    // 1. 定义整数类型常数操作ConstantIntOp
    def ConstantIntOp : Toy_Op<"constantint", [Pure]> {
      // Provide a summary and description for this operation. This can be used to
      // auto-generate documentation of the operations within our dialect.
      let summary = "integer constant";
      let description = [{
        ConstantInt operation turns a literal into an SSA value. The data is attached
        to the operation as an attribute. For example:

        ```mlir
          %0 = toy.constant dense<[[1, 2, 3], [4, 5, 6]]>
                            : tensor<2x3xi32>
        ```
      }];

      // The constant operation takes an attribute as the only input.
      let arguments = (ins I64ElementsAttr:$value);

      // The constant operation returns a single value of TensorType.
      let results = (outs I64Tensor);

      // Indicate that the operation has a custom parser and printer method.
      let hasCustomAssemblyFormat = 1;

      // Add custom build methods for the constant operation. These method populates
      // the `state` that MLIR uses to create operations, i.e. these are used when
      // using `builder.create<ConstantOp>(...)`.
      let builders = [
        // Build a constant with a given constant tensor value.
        OpBuilder<(ins "DenseIntElementsAttr":$value), [{
          build($_builder, $_state, value.getType(), value);
        }]>,

        // Build a constant with a given constant integral value.
        OpBuilder<(ins "int64_t":$value)>
      ];

      // Indicate that additional verification for this operation is necessary.
      let hasVerifier = 1;
    }

    def AndOp : Toy_Op<"and",
        [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {

      ...

      // 2. 修改AndOp的参数类型： F64Tensor -> I64Tensor
      let arguments = (ins I64Tensor:$lhs, I64Tensor:$rhs);
      let results = (outs I64Tensor);

      ...

    }

    def OrOp : Toy_Op<"or",
        [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {

      ...

      // 3. 修改OrOp的参数类型： F64Tensor -> I64Tensor
      let arguments = (ins I64Tensor:$lhs, I64Tensor:$rhs);
      let results = (outs I64Tensor);

      ...

    }

    def CastOp : Toy_Op<"cast", [
        DeclareOpInterfaceMethods<CastOpInterface>,
        DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
        Pure,
        SameOperandsAndResultShape
      ]> {

      ...
      // 4. 修改CastOp的参数使其兼容I64Tensor
      let arguments = (ins AnyTypeOf<[F64Tensor, I64Tensor]>:$input);
      let results   = (outs AnyTypeOf<[F64Tensor, I64Tensor]>:$output);

      ...

    }

    def GenericCallOp : Toy_Op<"generic_call",
    [DeclareOpInterfaceMethods<CallOpInterface>]> {

      ...
      // The generic call operation takes a symbol reference attribute as the
      // callee, and inputs for the call.
      let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<F64Tensor>:$inputs);

      // 5. 使GenricCallOp的返回值兼容I64Tensor
      // The generic call operation returns a single value of TensorType.
      let results = (outs AnyTypeOf<[F64Tensor, I64Tensor]>:$output);

      ...

    }

    def PrintOp : Toy_Op<"print"> {

      ...

      // 6. 增加PrintOp对I64类型的支持
      let arguments = (ins AnyTypeOf<[F64Tensor, F64MemRef, I64Tensor, I64MemRef]>:$input);

      ...

    }

    def ReshapeOp : Toy_Op<"reshape", [Pure]> {

      ...

      // 7.增加ReshapeOp对I64类型的支持
      let arguments = (ins AnyTypeOf<[F64Tensor, I64Tensor]>:$input);

      ...

      let results = (outs AnyTypeOf<[StaticShapeTensorOf<[F64]>, StaticShapeTensorOf<[I64]>]>);
    }
    ```

5. 在MLIRGen中实现对整数类型的支持

    ```c++
    // <MLIRGen.cpp>

    class MLIRGenImpl {

      ...

      mlir::Value mlirGen(LiteralExprAST &lit) {
        auto type = getType(lit.getDims());

        if (lit.getValueType() == type_integer)
          type = getIntType(lit.getDims());

        ValueType valueType = lit.getValueType();

        // The attribute is a vector with a floating point value per element
        // (number) in the array, see `collectData()` below for more details.
        std::vector<int64_t> intdata;
        std::vector<double> floatdata;

        // 1. 在生成Literal对应的MLIR时，增加对整数类型的判定已经相应的处理
        if (valueType == type_integer) {
          intdata.reserve(std::accumulate(lit.getDims().begin(), lit.getDims().end(), 1,
                                      std::multiplies<int>()));
          collectIntData(lit, intdata);

          // The type of this attribute is tensor of 64-bit floating-point with the
          // shape of the literal.
          mlir::Type elementType = builder.getI64Type();
          auto dataType = mlir::RankedTensorType::get(lit.getDims(), elementType);

          // This is the actual attribute that holds the list of values for this
          // tensor literal.
          auto dataAttribute =
              mlir::DenseIntElementsAttr::get(dataType, llvm::ArrayRef(intdata));

          // Build the MLIR op `toy.constantint`. This invokes the `ConstantIntOp::build`
          // method.
          return builder.create<ConstantIntOp>(loc(lit.loc()), type, dataAttribute);
        } else {
          floatdata.reserve(std::accumulate(lit.getDims().begin(), lit.getDims().end(), 1,
                                      std::multiplies<int>()));
          collectFloatData(lit, floatdata);

          // The type of this attribute is tensor of 64-bit floating-point with the
          // shape of the literal.
          mlir::Type elementType = builder.getF64Type();
          auto dataType = mlir::RankedTensorType::get(lit.getDims(), elementType);

          // This is the actual attribute that holds the list of values for this
          // tensor literal.
          auto dataAttribute =
              mlir::DenseElementsAttr::get(dataType, llvm::ArrayRef(floatdata));

          // Build the MLIR op `toy.constant`. This invokes the `ConstantOp::build`
          // method.
          return builder.create<ConstantOp>(loc(lit.loc()), type, dataAttribute);
        }
      }

      // 2. 将原有的collectData函数扩展为collectIntData和collectFloatData函数，分别用于处理整数数据和浮点数据
      void collectIntData(ExprAST &expr, std::vector<int64_t> &data) {
        if (auto *lit = dyn_cast<LiteralExprAST>(&expr)) {
          for (auto &value : lit->getValues())
            collectIntData(*value, data);
          return;
        }

        assert(isa<NumberExprAST>(expr) && "expected literal or number expr");
        data.push_back(cast<NumberExprAST>(expr).getIntValue());
      }

      void collectFloatData(ExprAST &expr, std::vector<double> &data) {
        if (auto *lit = dyn_cast<LiteralExprAST>(&expr)) {
          for (auto &value : lit->getValues())
            collectFloatData(*value, data);
          return;
        }

        assert(isa<NumberExprAST>(expr) && "expected literal or number expr");
        data.push_back(cast<NumberExprAST>(expr).getFloatValue());
      }

      ...

      // 3. 根据number类型创建相应的ConstantOp/ConstantIntOp
      mlir::Value mlirGen(NumberExprAST &num) {
        if (num.valueType == type_integer) {
          return builder.create<ConstantIntOp>(loc(num.loc()), num.getIntValue());
        } else if (num.valueType == type_float) {
          return builder.create<ConstantOp>(loc(num.loc()), num.getFloatValue());
        }
      }

      ...

      /// Handle an integer variable declaration, we'll codegen the expression that
      /// forms the initializer and record the value in the symbol table before
      /// returning it. Future expressions will be able to reference this variable
      /// through symbol table lookup.
      // 4. 增加对应整数类型声明语法的mlirGen函数
      mlir::Value mlirGen(VarIntDeclExprAST &varintdecl) {
        auto *init = varintdecl.getInitVal();
        if (!init) {
          emitError(loc(varintdecl.loc()),
                    "missing initializer in variable declaration");
          return nullptr;
        }

        mlir::Value value = mlirGen(*init);
        if (::llvm::DebugFlag)
          value.print(llvm::dbgs() << "\n");
        if (!value)
          return nullptr;

        // We have the initializer value, but in case the variable was declared
        // with specific shape, we emit a "reshape" operation. It will get
        // optimized out later as needed.
        if (!varintdecl.getType().shape.empty()) {
          value = builder.create<ReshapeOp>(loc(varintdecl.loc()),
                                            getIntType(varintdecl.getType()), value);
        }

        // Register the value in the symbol table.
        if (failed(declare(varintdecl.getName(), value)))
          return nullptr;
        return value;
      }

      ...

      /// Codegen a list of expression, return failure if one of them hit an error.
      mlir::LogicalResult mlirGen(ExprASTList &blockAST) {
        ScopedHashTableScope<StringRef, mlir::Value> varScope(symbolTable);
        for (auto &expr : blockAST) {

          ...

          // 5. 增加对VarIntDeclExprAST的处理
          if (auto *vardecl = dyn_cast<VarIntDeclExprAST>(expr.get())) {
            if (!mlirGen(*vardecl))
              return mlir::failure();
            continue;
          }

          ...

        }
      }

      ...

      mlir::Type getType(const VarType &type) { return getType(type.shape); }

      // 6. 增加getIntType()函数
      mlir::Type getIntType(ArrayRef<int64_t> shape) {
        // If the shape is empty, then this type is unranked.
        if (shape.empty())
          return mlir::UnrankedTensorType::get(builder.getI64Type());

        // Otherwise, we use the given shape.
        return mlir::RankedTensorType::get(shape, builder.getI64Type());
      }

      /// Build an MLIR type from a Toy AST variable type (forward to the generic
      /// getType above).
      mlir::Type getIntType(const VarType &type) { return getIntType(type.shape); }

    }
    ```

6. 在Dialect.cpp中声明ConstantIntOp

    ```c++
    //===----------------------------------------------------------------------===//
    // ConstantIntOp
    //===----------------------------------------------------------------------===//

    /// Build a constantint operation.
    /// The builder is passed as an argument, so is the state that this method is
    /// expected to fill in order to build the operation.
    void ConstantIntOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                          int64_t value) {
      auto dataType = RankedTensorType::get({}, builder.getI64Type());
      auto dataAttribute = DenseIntElementsAttr::get(dataType, value);
      ConstantIntOp::build(builder, state, dataType, dataAttribute);
    }

    /// The 'OpAsmParser' class provides a collection of methods for parsing
    /// various punctuation, as well as attributes, operands, types, etc. Each of
    /// these methods returns a `ParseResult`. This class is a wrapper around
    /// `LogicalResult` that can be converted to a boolean `true` value on failure,
    /// or `false` on success. This allows for easily chaining together a set of
    /// parser rules. These rules are used to populate an `mlir::OperationState`
    /// similarly to the `build` methods described above.
    mlir::ParseResult ConstantIntOp::parse(mlir::OpAsmParser &parser,
                                        mlir::OperationState &result) {
      mlir::DenseIntElementsAttr value;
      if (parser.parseOptionalAttrDict(result.attributes) ||
          parser.parseAttribute(value, "value", result.attributes))
        return failure();

      result.addTypes(value.getType());
      return success();
    }

    /// The 'OpAsmPrinter' class is a stream that allows for formatting
    /// strings, attributes, operands, types, etc.
    void ConstantIntOp::print(mlir::OpAsmPrinter &printer) {
      printer << " ";
      printer.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
      printer << getValue();
    }

    /// Verifier for the constant operation. This corresponds to the
    /// `let hasVerifier = 1` in the op definition.
    mlir::LogicalResult ConstantIntOp::verify() {
      // If the return type of the constant is not an unranked tensor, the shape
      // must match the shape of the attribute holding the data.
      auto resultType = llvm::dyn_cast<mlir::RankedTensorType>(getResult().getType());
      if (!resultType)
        return success();

      // Check that the rank of the attribute type matches the rank of the constant
      // result type.
      auto attrType = llvm::cast<mlir::RankedTensorType>(getValue().getType());
      if (attrType.getRank() != resultType.getRank()) {
        return emitOpError("return type must match the one of the attached value "
                          "attribute: ")
              << attrType.getRank() << " != " << resultType.getRank();
      }

      // Check that each of the dimensions match between the two types.
      for (int dim = 0, dimE = attrType.getRank(); dim < dimE; ++dim) {
        if (attrType.getShape()[dim] != resultType.getShape()[dim]) {
          return emitOpError(
                    "return type shape mismatches its attribute at dimension ")
                << dim << ": " << attrType.getShape()[dim]
                << " != " << resultType.getShape()[dim];
        }
      }
      return mlir::success();
    }
    ```

7. 修改ToyCombine使其兼容整数类型

    ```TableGen
    def ReshapeConstant :
      NativeCodeCall<"$0.reshape(::llvm::cast<ShapedType>($1.getType()))">;
    def ReshapeConstantInt :
      NativeCodeCall<"$0.reshape(::llvm::cast<ShapedType>($1.getType()))">;
    def FoldConstantReshapeOptPattern : Pat<
      (ReshapeOp:$res (ConstantOp $arg)),
      (ConstantOp (ReshapeConstant $arg, $res))
      >;
    def FoldConstantIntReshapeOptPattern : Pat<
      (ReshapeOp:$res (ConstantIntOp $arg)),
      (ConstantIntOp (ReshapeConstantInt $arg, $res))
      >;
    ```

    ```c++
    /// Register our patterns as "canonicalization" patterns on the ReshapeOp so
    /// that they can be picked up by the Canonicalization framework.
    void ReshapeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                MLIRContext *context) {
      results.add<ReshapeReshapeOptPattern, RedundantReshapeOptPattern,
                  FoldConstantReshapeOptPattern,
                  FoldConstantIntReshapeOptPattern>(context);
    }
    ```

8. 定义ConstantIntOpLowering使新增的Toy整数类型能够Lowering到Affine Dialect上

    ```c++
    //===----------------------------------------------------------------------===//
    // ToyToAffine RewritePatterns: ConstantInt operations
    //===----------------------------------------------------------------------===//

    struct ConstantIntOpLowering : public OpRewritePattern<toy::ConstantIntOp> {
      using OpRewritePattern<toy::ConstantIntOp>::OpRewritePattern;

      LogicalResult matchAndRewrite(toy::ConstantIntOp op,
                                    PatternRewriter &rewriter) const final {
        DenseIntElementsAttr constantintValue = op.getValue();
        Location loc = op.getLoc();

        // When lowering the constantint operation, we allocate and assign the constantint
        // values to a corresponding memref allocation.
        auto tensorType = llvm::cast<RankedTensorType>(op.getType());
        auto memRefType = convertTensorToMemRef(tensorType);
        auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

        // We will be generating constantint indices up-to the largest dimension.
        // Create these constantints up-front to avoid large amounts of redundant
        // operations.
        auto valueShape = memRefType.getShape();
        SmallVector<Value, 8> constantintIndices;

        if (!valueShape.empty()) {
          for (auto i : llvm::seq<int64_t>(
                  0, *std::max_element(valueShape.begin(), valueShape.end())))
            constantintIndices.push_back(
                rewriter.create<arith::ConstantIndexOp>(loc, i));
        } else {
          // This is the case of a tensor of rank 0.
          constantintIndices.push_back(
              rewriter.create<arith::ConstantIndexOp>(loc, 0));
        }

        // The constantint operation represents a multi-dimensional constantint, so we
        // will need to generate a store for each of the elements. The following
        // functor recursively walks the dimensions of the constantint shape,
        // generating a store when the recursion hits the base case.
        SmallVector<Value, 2> indices;
        auto valueIt = constantintValue.value_begin<IntegerAttr>();
        std::function<void(uint64_t)> storeElements = [&](uint64_t dimension) {
          // The last dimension is the base case of the recursion, at this point
          // we store the element at the given index.
          if (dimension == valueShape.size()) {
            rewriter.create<affine::AffineStoreOp>(
                loc, rewriter.create<arith::ConstantOp>(loc, *valueIt++), alloc,
                llvm::ArrayRef(indices));
            return;
          }

          // Otherwise, iterate over the current dimension and add the indices to
          // the list.
          for (uint64_t i = 0, e = valueShape[dimension]; i != e; ++i) {
            indices.push_back(constantintIndices[i]);
            storeElements(dimension + 1);
            indices.pop_back();
          }
        };

        // Start the element storing recursion from the first dimension.
        storeElements(/*dimension=*/0);

        // Replace this operation with the generated alloc.
        rewriter.replaceOp(op, alloc);
        return success();
      }
    };

    ...

    //在Lowering pattern中加上ConstantIntOpLowering
    patterns.add<AddOpLowering, SubOpLowering, MulOpLowering, AndOpLowering,
              OrOpLowering, ConstantOpLowering, ConstantIntOpLowering,
              FuncOpLowering, MulOpLowering, PrintOpLowering, ReturnOpLowering,
              TransposeOpLowering>(
    &getContext());
    ```

    至此，可以通过varint来定义整数类型变量,修改codegen.toy,引入
    
    ```
    def multiply_transpose(a, b) {
      return transpose(a) * transpose(b);
    }

    def main() {
      var a<2, 3> = [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]];
      var b<2, 3> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
      varint ia<2, 3> = [[7, 8, 9], [10, 11, 12]];
      varint ib<2, 3> = [13, 14, 15, 16, 17, 18];
      var c = multiply_transpose(a, b);
      var d = multiply_transpose(b, a);
      print(c);
      print(d);
      
      # new operations
      var e = ia | ib;
      var f = ia & ib;
      print(e);
      print(f);

      var g = a + b;
      var h = a - b;
      var i = a * b;
      print(g);
      print(h);
      print(i);
    }
    ```

    再次执行`./build/bin/toyc-ch6 -emit=mlir-affine ./toy_mod/src/Ch6/codegen.toy -opt`命令就可以生成lowering到affine dialect的mlir

    ```mlir
    module {
      func.func @main() {
        %c18_i64 = arith.constant 18 : i64
        %c17_i64 = arith.constant 17 : i64
        %c16_i64 = arith.constant 16 : i64
        %c15_i64 = arith.constant 15 : i64
        %c14_i64 = arith.constant 14 : i64
        %c13_i64 = arith.constant 13 : i64
        %c12_i64 = arith.constant 12 : i64
        %c11_i64 = arith.constant 11 : i64
        %c10_i64 = arith.constant 10 : i64
        %c9_i64 = arith.constant 9 : i64
        %c8_i64 = arith.constant 8 : i64
        %c7_i64 = arith.constant 7 : i64
        %cst = arith.constant 6.000000e+00 : f64
        %cst_0 = arith.constant 5.000000e+00 : f64
        %cst_1 = arith.constant 4.000000e+00 : f64
        %cst_2 = arith.constant 3.000000e+00 : f64
        %cst_3 = arith.constant 2.000000e+00 : f64
        %cst_4 = arith.constant 1.000000e+00 : f64
        %cst_5 = arith.constant 6.600000e+00 : f64
        %cst_6 = arith.constant 5.500000e+00 : f64
        %cst_7 = arith.constant 4.400000e+00 : f64
        %cst_8 = arith.constant 3.300000e+00 : f64
        %cst_9 = arith.constant 2.200000e+00 : f64
        %cst_10 = arith.constant 1.100000e+00 : f64
        %alloc = memref.alloc() : memref<2x3xf64>
        %alloc_11 = memref.alloc() : memref<2x3xf64>
        %alloc_12 = memref.alloc() : memref<2x3xf64>
        %alloc_13 = memref.alloc() : memref<2x3xi64>
        %alloc_14 = memref.alloc() : memref<2x3xi64>
        %alloc_15 = memref.alloc() : memref<3x2xf64>
        %alloc_16 = memref.alloc() : memref<3x2xf64>
        %alloc_17 = memref.alloc() : memref<2x3xi64>
        %alloc_18 = memref.alloc() : memref<2x3xi64>
        %alloc_19 = memref.alloc() : memref<2x3xf64>
        %alloc_20 = memref.alloc() : memref<2x3xf64>
        affine.store %cst_10, %alloc_20[0, 0] : memref<2x3xf64>
        affine.store %cst_9, %alloc_20[0, 1] : memref<2x3xf64>
        affine.store %cst_8, %alloc_20[0, 2] : memref<2x3xf64>
        affine.store %cst_7, %alloc_20[1, 0] : memref<2x3xf64>
        affine.store %cst_6, %alloc_20[1, 1] : memref<2x3xf64>
        affine.store %cst_5, %alloc_20[1, 2] : memref<2x3xf64>
        affine.store %cst_4, %alloc_19[0, 0] : memref<2x3xf64>
        affine.store %cst_3, %alloc_19[0, 1] : memref<2x3xf64>
        affine.store %cst_2, %alloc_19[0, 2] : memref<2x3xf64>
        affine.store %cst_1, %alloc_19[1, 0] : memref<2x3xf64>
        affine.store %cst_0, %alloc_19[1, 1] : memref<2x3xf64>
        affine.store %cst, %alloc_19[1, 2] : memref<2x3xf64>
        affine.store %c7_i64, %alloc_18[0, 0] : memref<2x3xi64>
        affine.store %c8_i64, %alloc_18[0, 1] : memref<2x3xi64>
        affine.store %c9_i64, %alloc_18[0, 2] : memref<2x3xi64>
        affine.store %c10_i64, %alloc_18[1, 0] : memref<2x3xi64>
        affine.store %c11_i64, %alloc_18[1, 1] : memref<2x3xi64>
        affine.store %c12_i64, %alloc_18[1, 2] : memref<2x3xi64>
        affine.store %c13_i64, %alloc_17[0, 0] : memref<2x3xi64>
        affine.store %c14_i64, %alloc_17[0, 1] : memref<2x3xi64>
        affine.store %c15_i64, %alloc_17[0, 2] : memref<2x3xi64>
        affine.store %c16_i64, %alloc_17[1, 0] : memref<2x3xi64>
        affine.store %c17_i64, %alloc_17[1, 1] : memref<2x3xi64>
        affine.store %c18_i64, %alloc_17[1, 2] : memref<2x3xi64>
        affine.for %arg0 = 0 to 3 {
          affine.for %arg1 = 0 to 2 {
            %0 = affine.load %alloc_20[%arg1, %arg0] : memref<2x3xf64>
            %1 = affine.load %alloc_19[%arg1, %arg0] : memref<2x3xf64>
            %2 = arith.mulf %0, %1 : f64
            affine.store %2, %alloc_16[%arg0, %arg1] : memref<3x2xf64>
            %3 = arith.mulf %1, %0 : f64
            affine.store %3, %alloc_15[%arg0, %arg1] : memref<3x2xf64>
          }
        }
        toy.print %alloc_16 : memref<3x2xf64>
        toy.print %alloc_15 : memref<3x2xf64>
        affine.for %arg0 = 0 to 2 {
          affine.for %arg1 = 0 to 3 {
            %0 = affine.load %alloc_18[%arg0, %arg1] : memref<2x3xi64>
            %1 = affine.load %alloc_17[%arg0, %arg1] : memref<2x3xi64>
            %2 = arith.ori %0, %1 : i64
            affine.store %2, %alloc_14[%arg0, %arg1] : memref<2x3xi64>
          }
        }
        affine.for %arg0 = 0 to 2 {
          affine.for %arg1 = 0 to 3 {
            %0 = affine.load %alloc_18[%arg0, %arg1] : memref<2x3xi64>
            %1 = affine.load %alloc_17[%arg0, %arg1] : memref<2x3xi64>
            %2 = arith.andi %0, %1 : i64
            affine.store %2, %alloc_13[%arg0, %arg1] : memref<2x3xi64>
          }
        }
        toy.print %alloc_14 : memref<2x3xi64>
        toy.print %alloc_13 : memref<2x3xi64>
        affine.for %arg0 = 0 to 2 {
          affine.for %arg1 = 0 to 3 {
            %0 = affine.load %alloc_20[%arg0, %arg1] : memref<2x3xf64>
            %1 = affine.load %alloc_19[%arg0, %arg1] : memref<2x3xf64>
            %2 = arith.addf %0, %1 : f64
            affine.store %2, %alloc_12[%arg0, %arg1] : memref<2x3xf64>
          }
        }
        affine.for %arg0 = 0 to 2 {
          affine.for %arg1 = 0 to 3 {
            %0 = affine.load %alloc_20[%arg0, %arg1] : memref<2x3xf64>
            %1 = affine.load %alloc_19[%arg0, %arg1] : memref<2x3xf64>
            %2 = arith.subf %0, %1 : f64
            affine.store %2, %alloc_11[%arg0, %arg1] : memref<2x3xf64>
          }
        }
        affine.for %arg0 = 0 to 2 {
          affine.for %arg1 = 0 to 3 {
            %0 = affine.load %alloc_20[%arg0, %arg1] : memref<2x3xf64>
            %1 = affine.load %alloc_19[%arg0, %arg1] : memref<2x3xf64>
            %2 = arith.mulf %0, %1 : f64
            affine.store %2, %alloc[%arg0, %arg1] : memref<2x3xf64>
          }
        }
        toy.print %alloc_12 : memref<2x3xf64>
        toy.print %alloc_11 : memref<2x3xf64>
        toy.print %alloc : memref<2x3xf64>
        memref.dealloc %alloc_20 : memref<2x3xf64>
        memref.dealloc %alloc_19 : memref<2x3xf64>
        memref.dealloc %alloc_18 : memref<2x3xi64>
        memref.dealloc %alloc_17 : memref<2x3xi64>
        memref.dealloc %alloc_16 : memref<3x2xf64>
        memref.dealloc %alloc_15 : memref<3x2xf64>
        memref.dealloc %alloc_14 : memref<2x3xi64>
        memref.dealloc %alloc_13 : memref<2x3xi64>
        memref.dealloc %alloc_12 : memref<2x3xf64>
        memref.dealloc %alloc_11 : memref<2x3xf64>
        memref.dealloc %alloc : memref<2x3xf64>
        return
      }
    }
    ```

#### 将PrintOp lowering到LLVM Dialect

1. Toy原版的Example里已经提供了将PrintOp Lowering到LLVM的代码，此时执行`./build/bin/toyc-ch6 -emit=mlir-llvm ./toy_mod/src/Ch6/codegen.toy -opt`命令就可以生成lowering到llvm dialect的mlir

    ```mlir
      module {
        llvm.func @free(!llvm.ptr)
        llvm.mlir.global internal constant @nl("\0A\00") {addr_space = 0 : i32}
        llvm.mlir.global internal constant @frmt_spec("%f \00") {addr_space = 0 : i32}
        llvm.func @printf(!llvm.ptr<i8>, ...) -> i32
        llvm.func @malloc(i64) -> !llvm.ptr
        llvm.func @main() {
          %0 = llvm.mlir.constant(18 : i64) : i64
          %1 = llvm.mlir.constant(17 : i64) : i64
          %2 = llvm.mlir.constant(16 : i64) : i64
          %3 = llvm.mlir.constant(15 : i64) : i64
          %4 = llvm.mlir.constant(14 : i64) : i64
          %5 = llvm.mlir.constant(13 : i64) : i64
          %6 = llvm.mlir.constant(12 : i64) : i64
          %7 = llvm.mlir.constant(11 : i64) : i64
          %8 = llvm.mlir.constant(10 : i64) : i64
          %9 = llvm.mlir.constant(9 : i64) : i64
          %10 = llvm.mlir.constant(8 : i64) : i64
          %11 = llvm.mlir.constant(7 : i64) : i64
          %12 = llvm.mlir.constant(6.000000e+00 : f64) : f64
          %13 = llvm.mlir.constant(5.000000e+00 : f64) : f64
          %14 = llvm.mlir.constant(4.000000e+00 : f64) : f64
          %15 = llvm.mlir.constant(3.000000e+00 : f64) : f64
          %16 = llvm.mlir.constant(2.000000e+00 : f64) : f64
          %17 = llvm.mlir.constant(1.000000e+00 : f64) : f64
          %18 = llvm.mlir.constant(6.600000e+00 : f64) : f64
          %19 = llvm.mlir.constant(5.500000e+00 : f64) : f64
          %20 = llvm.mlir.constant(4.400000e+00 : f64) : f64
          %21 = llvm.mlir.constant(3.300000e+00 : f64) : f64
          %22 = llvm.mlir.constant(2.200000e+00 : f64) : f64
          %23 = llvm.mlir.constant(1.100000e+00 : f64) : f64
          %24 = llvm.mlir.constant(2 : index) : i64
          %25 = llvm.mlir.constant(3 : index) : i64
          %26 = llvm.mlir.constant(1 : index) : i64
          %27 = llvm.mlir.constant(6 : index) : i64
          %28 = llvm.mlir.null : !llvm.ptr
          %29 = llvm.getelementptr %28[%27] : (!llvm.ptr, i64) -> !llvm.ptr, f64
          %30 = llvm.ptrtoint %29 : !llvm.ptr to i64
          %31 = llvm.call @malloc(%30) : (i64) -> !llvm.ptr
          %32 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
          %33 = llvm.insertvalue %31, %32[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %34 = llvm.insertvalue %31, %33[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %35 = llvm.mlir.constant(0 : index) : i64
          %36 = llvm.insertvalue %35, %34[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %37 = llvm.insertvalue %24, %36[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %38 = llvm.insertvalue %25, %37[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %39 = llvm.insertvalue %25, %38[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %40 = llvm.insertvalue %26, %39[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %41 = llvm.mlir.constant(2 : index) : i64
          %42 = llvm.mlir.constant(3 : index) : i64
          %43 = llvm.mlir.constant(1 : index) : i64
          %44 = llvm.mlir.constant(6 : index) : i64
          %45 = llvm.mlir.null : !llvm.ptr
          %46 = llvm.getelementptr %45[%44] : (!llvm.ptr, i64) -> !llvm.ptr, f64
          %47 = llvm.ptrtoint %46 : !llvm.ptr to i64
          %48 = llvm.call @malloc(%47) : (i64) -> !llvm.ptr
          %49 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
          %50 = llvm.insertvalue %48, %49[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %51 = llvm.insertvalue %48, %50[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %52 = llvm.mlir.constant(0 : index) : i64
          %53 = llvm.insertvalue %52, %51[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %54 = llvm.insertvalue %41, %53[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %55 = llvm.insertvalue %42, %54[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %56 = llvm.insertvalue %42, %55[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %57 = llvm.insertvalue %43, %56[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %58 = llvm.mlir.constant(2 : index) : i64
          %59 = llvm.mlir.constant(3 : index) : i64
          %60 = llvm.mlir.constant(1 : index) : i64
          %61 = llvm.mlir.constant(6 : index) : i64
          %62 = llvm.mlir.null : !llvm.ptr
          %63 = llvm.getelementptr %62[%61] : (!llvm.ptr, i64) -> !llvm.ptr, f64
          %64 = llvm.ptrtoint %63 : !llvm.ptr to i64
          %65 = llvm.call @malloc(%64) : (i64) -> !llvm.ptr
          %66 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
          %67 = llvm.insertvalue %65, %66[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %68 = llvm.insertvalue %65, %67[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %69 = llvm.mlir.constant(0 : index) : i64
          %70 = llvm.insertvalue %69, %68[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %71 = llvm.insertvalue %58, %70[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %72 = llvm.insertvalue %59, %71[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %73 = llvm.insertvalue %59, %72[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %74 = llvm.insertvalue %60, %73[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %75 = llvm.mlir.constant(2 : index) : i64
          %76 = llvm.mlir.constant(3 : index) : i64
          %77 = llvm.mlir.constant(1 : index) : i64
          %78 = llvm.mlir.constant(6 : index) : i64
          %79 = llvm.mlir.null : !llvm.ptr
          %80 = llvm.getelementptr %79[%78] : (!llvm.ptr, i64) -> !llvm.ptr, i64
          %81 = llvm.ptrtoint %80 : !llvm.ptr to i64
          %82 = llvm.call @malloc(%81) : (i64) -> !llvm.ptr
          %83 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
          %84 = llvm.insertvalue %82, %83[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %85 = llvm.insertvalue %82, %84[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %86 = llvm.mlir.constant(0 : index) : i64
          %87 = llvm.insertvalue %86, %85[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %88 = llvm.insertvalue %75, %87[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %89 = llvm.insertvalue %76, %88[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %90 = llvm.insertvalue %76, %89[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %91 = llvm.insertvalue %77, %90[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %92 = llvm.mlir.constant(2 : index) : i64
          %93 = llvm.mlir.constant(3 : index) : i64
          %94 = llvm.mlir.constant(1 : index) : i64
          %95 = llvm.mlir.constant(6 : index) : i64
          %96 = llvm.mlir.null : !llvm.ptr
          %97 = llvm.getelementptr %96[%95] : (!llvm.ptr, i64) -> !llvm.ptr, i64
          %98 = llvm.ptrtoint %97 : !llvm.ptr to i64
          %99 = llvm.call @malloc(%98) : (i64) -> !llvm.ptr
          %100 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
          %101 = llvm.insertvalue %99, %100[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %102 = llvm.insertvalue %99, %101[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %103 = llvm.mlir.constant(0 : index) : i64
          %104 = llvm.insertvalue %103, %102[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %105 = llvm.insertvalue %92, %104[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %106 = llvm.insertvalue %93, %105[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %107 = llvm.insertvalue %93, %106[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %108 = llvm.insertvalue %94, %107[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %109 = llvm.mlir.constant(3 : index) : i64
          %110 = llvm.mlir.constant(2 : index) : i64
          %111 = llvm.mlir.constant(1 : index) : i64
          %112 = llvm.mlir.constant(6 : index) : i64
          %113 = llvm.mlir.null : !llvm.ptr
          %114 = llvm.getelementptr %113[%112] : (!llvm.ptr, i64) -> !llvm.ptr, f64
          %115 = llvm.ptrtoint %114 : !llvm.ptr to i64
          %116 = llvm.call @malloc(%115) : (i64) -> !llvm.ptr
          %117 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
          %118 = llvm.insertvalue %116, %117[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %119 = llvm.insertvalue %116, %118[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %120 = llvm.mlir.constant(0 : index) : i64
          %121 = llvm.insertvalue %120, %119[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %122 = llvm.insertvalue %109, %121[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %123 = llvm.insertvalue %110, %122[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %124 = llvm.insertvalue %110, %123[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %125 = llvm.insertvalue %111, %124[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %126 = llvm.mlir.constant(3 : index) : i64
          %127 = llvm.mlir.constant(2 : index) : i64
          %128 = llvm.mlir.constant(1 : index) : i64
          %129 = llvm.mlir.constant(6 : index) : i64
          %130 = llvm.mlir.null : !llvm.ptr
          %131 = llvm.getelementptr %130[%129] : (!llvm.ptr, i64) -> !llvm.ptr, f64
          %132 = llvm.ptrtoint %131 : !llvm.ptr to i64
          %133 = llvm.call @malloc(%132) : (i64) -> !llvm.ptr
          %134 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
          %135 = llvm.insertvalue %133, %134[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %136 = llvm.insertvalue %133, %135[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %137 = llvm.mlir.constant(0 : index) : i64
          %138 = llvm.insertvalue %137, %136[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %139 = llvm.insertvalue %126, %138[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %140 = llvm.insertvalue %127, %139[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %141 = llvm.insertvalue %127, %140[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %142 = llvm.insertvalue %128, %141[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %143 = llvm.mlir.constant(2 : index) : i64
          %144 = llvm.mlir.constant(3 : index) : i64
          %145 = llvm.mlir.constant(1 : index) : i64
          %146 = llvm.mlir.constant(6 : index) : i64
          %147 = llvm.mlir.null : !llvm.ptr
          %148 = llvm.getelementptr %147[%146] : (!llvm.ptr, i64) -> !llvm.ptr, i64
          %149 = llvm.ptrtoint %148 : !llvm.ptr to i64
          %150 = llvm.call @malloc(%149) : (i64) -> !llvm.ptr
          %151 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
          %152 = llvm.insertvalue %150, %151[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %153 = llvm.insertvalue %150, %152[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %154 = llvm.mlir.constant(0 : index) : i64
          %155 = llvm.insertvalue %154, %153[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %156 = llvm.insertvalue %143, %155[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %157 = llvm.insertvalue %144, %156[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %158 = llvm.insertvalue %144, %157[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %159 = llvm.insertvalue %145, %158[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %160 = llvm.mlir.constant(2 : index) : i64
          %161 = llvm.mlir.constant(3 : index) : i64
          %162 = llvm.mlir.constant(1 : index) : i64
          %163 = llvm.mlir.constant(6 : index) : i64
          %164 = llvm.mlir.null : !llvm.ptr
          %165 = llvm.getelementptr %164[%163] : (!llvm.ptr, i64) -> !llvm.ptr, i64
          %166 = llvm.ptrtoint %165 : !llvm.ptr to i64
          %167 = llvm.call @malloc(%166) : (i64) -> !llvm.ptr
          %168 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
          %169 = llvm.insertvalue %167, %168[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %170 = llvm.insertvalue %167, %169[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %171 = llvm.mlir.constant(0 : index) : i64
          %172 = llvm.insertvalue %171, %170[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %173 = llvm.insertvalue %160, %172[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %174 = llvm.insertvalue %161, %173[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %175 = llvm.insertvalue %161, %174[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %176 = llvm.insertvalue %162, %175[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %177 = llvm.mlir.constant(2 : index) : i64
          %178 = llvm.mlir.constant(3 : index) : i64
          %179 = llvm.mlir.constant(1 : index) : i64
          %180 = llvm.mlir.constant(6 : index) : i64
          %181 = llvm.mlir.null : !llvm.ptr
          %182 = llvm.getelementptr %181[%180] : (!llvm.ptr, i64) -> !llvm.ptr, f64
          %183 = llvm.ptrtoint %182 : !llvm.ptr to i64
          %184 = llvm.call @malloc(%183) : (i64) -> !llvm.ptr
          %185 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
          %186 = llvm.insertvalue %184, %185[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %187 = llvm.insertvalue %184, %186[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %188 = llvm.mlir.constant(0 : index) : i64
          %189 = llvm.insertvalue %188, %187[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %190 = llvm.insertvalue %177, %189[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %191 = llvm.insertvalue %178, %190[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %192 = llvm.insertvalue %178, %191[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %193 = llvm.insertvalue %179, %192[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %194 = llvm.mlir.constant(2 : index) : i64
          %195 = llvm.mlir.constant(3 : index) : i64
          %196 = llvm.mlir.constant(1 : index) : i64
          %197 = llvm.mlir.constant(6 : index) : i64
          %198 = llvm.mlir.null : !llvm.ptr
          %199 = llvm.getelementptr %198[%197] : (!llvm.ptr, i64) -> !llvm.ptr, f64
          %200 = llvm.ptrtoint %199 : !llvm.ptr to i64
          %201 = llvm.call @malloc(%200) : (i64) -> !llvm.ptr
          %202 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
          %203 = llvm.insertvalue %201, %202[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %204 = llvm.insertvalue %201, %203[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %205 = llvm.mlir.constant(0 : index) : i64
          %206 = llvm.insertvalue %205, %204[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %207 = llvm.insertvalue %194, %206[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %208 = llvm.insertvalue %195, %207[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %209 = llvm.insertvalue %195, %208[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %210 = llvm.insertvalue %196, %209[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %211 = llvm.mlir.constant(0 : index) : i64
          %212 = llvm.mlir.constant(0 : index) : i64
          %213 = llvm.extractvalue %210[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %214 = llvm.mlir.constant(3 : index) : i64
          %215 = llvm.mul %211, %214  : i64
          %216 = llvm.add %215, %212  : i64
          %217 = llvm.getelementptr %213[%216] : (!llvm.ptr, i64) -> !llvm.ptr, f64
          llvm.store %23, %217 : f64, !llvm.ptr
          %218 = llvm.mlir.constant(0 : index) : i64
          %219 = llvm.mlir.constant(1 : index) : i64
          %220 = llvm.extractvalue %210[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %221 = llvm.mlir.constant(3 : index) : i64
          %222 = llvm.mul %218, %221  : i64
          %223 = llvm.add %222, %219  : i64
          %224 = llvm.getelementptr %220[%223] : (!llvm.ptr, i64) -> !llvm.ptr, f64
          llvm.store %22, %224 : f64, !llvm.ptr
          %225 = llvm.mlir.constant(0 : index) : i64
          %226 = llvm.mlir.constant(2 : index) : i64
          %227 = llvm.extractvalue %210[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %228 = llvm.mlir.constant(3 : index) : i64
          %229 = llvm.mul %225, %228  : i64
          %230 = llvm.add %229, %226  : i64
          %231 = llvm.getelementptr %227[%230] : (!llvm.ptr, i64) -> !llvm.ptr, f64
          llvm.store %21, %231 : f64, !llvm.ptr
          %232 = llvm.mlir.constant(1 : index) : i64
          %233 = llvm.mlir.constant(0 : index) : i64
          %234 = llvm.extractvalue %210[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %235 = llvm.mlir.constant(3 : index) : i64
          %236 = llvm.mul %232, %235  : i64
          %237 = llvm.add %236, %233  : i64
          %238 = llvm.getelementptr %234[%237] : (!llvm.ptr, i64) -> !llvm.ptr, f64
          llvm.store %20, %238 : f64, !llvm.ptr
          %239 = llvm.mlir.constant(1 : index) : i64
          %240 = llvm.mlir.constant(1 : index) : i64
          %241 = llvm.extractvalue %210[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %242 = llvm.mlir.constant(3 : index) : i64
          %243 = llvm.mul %239, %242  : i64
          %244 = llvm.add %243, %240  : i64
          %245 = llvm.getelementptr %241[%244] : (!llvm.ptr, i64) -> !llvm.ptr, f64
          llvm.store %19, %245 : f64, !llvm.ptr
          %246 = llvm.mlir.constant(1 : index) : i64
          %247 = llvm.mlir.constant(2 : index) : i64
          %248 = llvm.extractvalue %210[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %249 = llvm.mlir.constant(3 : index) : i64
          %250 = llvm.mul %246, %249  : i64
          %251 = llvm.add %250, %247  : i64
          %252 = llvm.getelementptr %248[%251] : (!llvm.ptr, i64) -> !llvm.ptr, f64
          llvm.store %18, %252 : f64, !llvm.ptr
          %253 = llvm.mlir.constant(0 : index) : i64
          %254 = llvm.mlir.constant(0 : index) : i64
          %255 = llvm.extractvalue %193[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %256 = llvm.mlir.constant(3 : index) : i64
          %257 = llvm.mul %253, %256  : i64
          %258 = llvm.add %257, %254  : i64
          %259 = llvm.getelementptr %255[%258] : (!llvm.ptr, i64) -> !llvm.ptr, f64
          llvm.store %17, %259 : f64, !llvm.ptr
          %260 = llvm.mlir.constant(0 : index) : i64
          %261 = llvm.mlir.constant(1 : index) : i64
          %262 = llvm.extractvalue %193[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %263 = llvm.mlir.constant(3 : index) : i64
          %264 = llvm.mul %260, %263  : i64
          %265 = llvm.add %264, %261  : i64
          %266 = llvm.getelementptr %262[%265] : (!llvm.ptr, i64) -> !llvm.ptr, f64
          llvm.store %16, %266 : f64, !llvm.ptr
          %267 = llvm.mlir.constant(0 : index) : i64
          %268 = llvm.mlir.constant(2 : index) : i64
          %269 = llvm.extractvalue %193[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %270 = llvm.mlir.constant(3 : index) : i64
          %271 = llvm.mul %267, %270  : i64
          %272 = llvm.add %271, %268  : i64
          %273 = llvm.getelementptr %269[%272] : (!llvm.ptr, i64) -> !llvm.ptr, f64
          llvm.store %15, %273 : f64, !llvm.ptr
          %274 = llvm.mlir.constant(1 : index) : i64
          %275 = llvm.mlir.constant(0 : index) : i64
          %276 = llvm.extractvalue %193[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %277 = llvm.mlir.constant(3 : index) : i64
          %278 = llvm.mul %274, %277  : i64
          %279 = llvm.add %278, %275  : i64
          %280 = llvm.getelementptr %276[%279] : (!llvm.ptr, i64) -> !llvm.ptr, f64
          llvm.store %14, %280 : f64, !llvm.ptr
          %281 = llvm.mlir.constant(1 : index) : i64
          %282 = llvm.mlir.constant(1 : index) : i64
          %283 = llvm.extractvalue %193[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %284 = llvm.mlir.constant(3 : index) : i64
          %285 = llvm.mul %281, %284  : i64
          %286 = llvm.add %285, %282  : i64
          %287 = llvm.getelementptr %283[%286] : (!llvm.ptr, i64) -> !llvm.ptr, f64
          llvm.store %13, %287 : f64, !llvm.ptr
          %288 = llvm.mlir.constant(1 : index) : i64
          %289 = llvm.mlir.constant(2 : index) : i64
          %290 = llvm.extractvalue %193[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %291 = llvm.mlir.constant(3 : index) : i64
          %292 = llvm.mul %288, %291  : i64
          %293 = llvm.add %292, %289  : i64
          %294 = llvm.getelementptr %290[%293] : (!llvm.ptr, i64) -> !llvm.ptr, f64
          llvm.store %12, %294 : f64, !llvm.ptr
          %295 = llvm.mlir.constant(0 : index) : i64
          %296 = llvm.mlir.constant(0 : index) : i64
          %297 = llvm.extractvalue %176[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %298 = llvm.mlir.constant(3 : index) : i64
          %299 = llvm.mul %295, %298  : i64
          %300 = llvm.add %299, %296  : i64
          %301 = llvm.getelementptr %297[%300] : (!llvm.ptr, i64) -> !llvm.ptr, i64
          llvm.store %11, %301 : i64, !llvm.ptr
          %302 = llvm.mlir.constant(0 : index) : i64
          %303 = llvm.mlir.constant(1 : index) : i64
          %304 = llvm.extractvalue %176[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %305 = llvm.mlir.constant(3 : index) : i64
          %306 = llvm.mul %302, %305  : i64
          %307 = llvm.add %306, %303  : i64
          %308 = llvm.getelementptr %304[%307] : (!llvm.ptr, i64) -> !llvm.ptr, i64
          llvm.store %10, %308 : i64, !llvm.ptr
          %309 = llvm.mlir.constant(0 : index) : i64
          %310 = llvm.mlir.constant(2 : index) : i64
          %311 = llvm.extractvalue %176[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %312 = llvm.mlir.constant(3 : index) : i64
          %313 = llvm.mul %309, %312  : i64
          %314 = llvm.add %313, %310  : i64
          %315 = llvm.getelementptr %311[%314] : (!llvm.ptr, i64) -> !llvm.ptr, i64
          llvm.store %9, %315 : i64, !llvm.ptr
          %316 = llvm.mlir.constant(1 : index) : i64
          %317 = llvm.mlir.constant(0 : index) : i64
          %318 = llvm.extractvalue %176[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %319 = llvm.mlir.constant(3 : index) : i64
          %320 = llvm.mul %316, %319  : i64
          %321 = llvm.add %320, %317  : i64
          %322 = llvm.getelementptr %318[%321] : (!llvm.ptr, i64) -> !llvm.ptr, i64
          llvm.store %8, %322 : i64, !llvm.ptr
          %323 = llvm.mlir.constant(1 : index) : i64
          %324 = llvm.mlir.constant(1 : index) : i64
          %325 = llvm.extractvalue %176[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %326 = llvm.mlir.constant(3 : index) : i64
          %327 = llvm.mul %323, %326  : i64
          %328 = llvm.add %327, %324  : i64
          %329 = llvm.getelementptr %325[%328] : (!llvm.ptr, i64) -> !llvm.ptr, i64
          llvm.store %7, %329 : i64, !llvm.ptr
          %330 = llvm.mlir.constant(1 : index) : i64
          %331 = llvm.mlir.constant(2 : index) : i64
          %332 = llvm.extractvalue %176[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %333 = llvm.mlir.constant(3 : index) : i64
          %334 = llvm.mul %330, %333  : i64
          %335 = llvm.add %334, %331  : i64
          %336 = llvm.getelementptr %332[%335] : (!llvm.ptr, i64) -> !llvm.ptr, i64
          llvm.store %6, %336 : i64, !llvm.ptr
          %337 = llvm.mlir.constant(0 : index) : i64
          %338 = llvm.mlir.constant(0 : index) : i64
          %339 = llvm.extractvalue %159[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %340 = llvm.mlir.constant(3 : index) : i64
          %341 = llvm.mul %337, %340  : i64
          %342 = llvm.add %341, %338  : i64
          %343 = llvm.getelementptr %339[%342] : (!llvm.ptr, i64) -> !llvm.ptr, i64
          llvm.store %5, %343 : i64, !llvm.ptr
          %344 = llvm.mlir.constant(0 : index) : i64
          %345 = llvm.mlir.constant(1 : index) : i64
          %346 = llvm.extractvalue %159[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %347 = llvm.mlir.constant(3 : index) : i64
          %348 = llvm.mul %344, %347  : i64
          %349 = llvm.add %348, %345  : i64
          %350 = llvm.getelementptr %346[%349] : (!llvm.ptr, i64) -> !llvm.ptr, i64
          llvm.store %4, %350 : i64, !llvm.ptr
          %351 = llvm.mlir.constant(0 : index) : i64
          %352 = llvm.mlir.constant(2 : index) : i64
          %353 = llvm.extractvalue %159[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %354 = llvm.mlir.constant(3 : index) : i64
          %355 = llvm.mul %351, %354  : i64
          %356 = llvm.add %355, %352  : i64
          %357 = llvm.getelementptr %353[%356] : (!llvm.ptr, i64) -> !llvm.ptr, i64
          llvm.store %3, %357 : i64, !llvm.ptr
          %358 = llvm.mlir.constant(1 : index) : i64
          %359 = llvm.mlir.constant(0 : index) : i64
          %360 = llvm.extractvalue %159[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %361 = llvm.mlir.constant(3 : index) : i64
          %362 = llvm.mul %358, %361  : i64
          %363 = llvm.add %362, %359  : i64
          %364 = llvm.getelementptr %360[%363] : (!llvm.ptr, i64) -> !llvm.ptr, i64
          llvm.store %2, %364 : i64, !llvm.ptr
          %365 = llvm.mlir.constant(1 : index) : i64
          %366 = llvm.mlir.constant(1 : index) : i64
          %367 = llvm.extractvalue %159[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %368 = llvm.mlir.constant(3 : index) : i64
          %369 = llvm.mul %365, %368  : i64
          %370 = llvm.add %369, %366  : i64
          %371 = llvm.getelementptr %367[%370] : (!llvm.ptr, i64) -> !llvm.ptr, i64
          llvm.store %1, %371 : i64, !llvm.ptr
          %372 = llvm.mlir.constant(1 : index) : i64
          %373 = llvm.mlir.constant(2 : index) : i64
          %374 = llvm.extractvalue %159[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %375 = llvm.mlir.constant(3 : index) : i64
          %376 = llvm.mul %372, %375  : i64
          %377 = llvm.add %376, %373  : i64
          %378 = llvm.getelementptr %374[%377] : (!llvm.ptr, i64) -> !llvm.ptr, i64
          llvm.store %0, %378 : i64, !llvm.ptr
          %379 = llvm.mlir.constant(0 : index) : i64
          %380 = llvm.mlir.constant(3 : index) : i64
          %381 = llvm.mlir.constant(1 : index) : i64
          llvm.br ^bb1(%379 : i64)
        ^bb1(%382: i64):  // 2 preds: ^bb0, ^bb5
          %383 = llvm.icmp "slt" %382, %380 : i64
          llvm.cond_br %383, ^bb2, ^bb6
        ^bb2:  // pred: ^bb1
          %384 = llvm.mlir.constant(0 : index) : i64
          %385 = llvm.mlir.constant(2 : index) : i64
          %386 = llvm.mlir.constant(1 : index) : i64
          llvm.br ^bb3(%384 : i64)
        ^bb3(%387: i64):  // 2 preds: ^bb2, ^bb4
          %388 = llvm.icmp "slt" %387, %385 : i64
          llvm.cond_br %388, ^bb4, ^bb5
        ^bb4:  // pred: ^bb3
          %389 = llvm.extractvalue %210[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %390 = llvm.mlir.constant(3 : index) : i64
          %391 = llvm.mul %387, %390  : i64
          %392 = llvm.add %391, %382  : i64
          %393 = llvm.getelementptr %389[%392] : (!llvm.ptr, i64) -> !llvm.ptr, f64
          %394 = llvm.load %393 : !llvm.ptr -> f64
          %395 = llvm.extractvalue %193[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %396 = llvm.mlir.constant(3 : index) : i64
          %397 = llvm.mul %387, %396  : i64
          %398 = llvm.add %397, %382  : i64
          %399 = llvm.getelementptr %395[%398] : (!llvm.ptr, i64) -> !llvm.ptr, f64
          %400 = llvm.load %399 : !llvm.ptr -> f64
          %401 = llvm.fmul %394, %400  : f64
          %402 = llvm.extractvalue %142[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %403 = llvm.mlir.constant(2 : index) : i64
          %404 = llvm.mul %382, %403  : i64
          %405 = llvm.add %404, %387  : i64
          %406 = llvm.getelementptr %402[%405] : (!llvm.ptr, i64) -> !llvm.ptr, f64
          llvm.store %401, %406 : f64, !llvm.ptr
          %407 = llvm.fmul %400, %394  : f64
          %408 = llvm.extractvalue %125[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %409 = llvm.mlir.constant(2 : index) : i64
          %410 = llvm.mul %382, %409  : i64
          %411 = llvm.add %410, %387  : i64
          %412 = llvm.getelementptr %408[%411] : (!llvm.ptr, i64) -> !llvm.ptr, f64
          llvm.store %407, %412 : f64, !llvm.ptr
          %413 = llvm.add %387, %386  : i64
          llvm.br ^bb3(%413 : i64)
        ^bb5:  // pred: ^bb3
          %414 = llvm.add %382, %381  : i64
          llvm.br ^bb1(%414 : i64)
        ^bb6:  // pred: ^bb1
          %415 = llvm.mlir.addressof @frmt_spec : !llvm.ptr<array<4 x i8>>
          %416 = llvm.mlir.constant(0 : index) : i64
          %417 = llvm.getelementptr %415[%416, %416] : (!llvm.ptr<array<4 x i8>>, i64, i64) -> !llvm.ptr<i8>
          %418 = llvm.mlir.addressof @nl : !llvm.ptr<array<2 x i8>>
          %419 = llvm.mlir.constant(0 : index) : i64
          %420 = llvm.getelementptr %418[%419, %419] : (!llvm.ptr<array<2 x i8>>, i64, i64) -> !llvm.ptr<i8>
          %421 = llvm.mlir.constant(0 : index) : i64
          %422 = llvm.mlir.constant(3 : index) : i64
          %423 = llvm.mlir.constant(1 : index) : i64
          llvm.br ^bb7(%421 : i64)
        ^bb7(%424: i64):  // 2 preds: ^bb6, ^bb11
          %425 = llvm.icmp "slt" %424, %422 : i64
          llvm.cond_br %425, ^bb8, ^bb12
        ^bb8:  // pred: ^bb7
          %426 = llvm.mlir.constant(0 : index) : i64
          %427 = llvm.mlir.constant(2 : index) : i64
          %428 = llvm.mlir.constant(1 : index) : i64
          llvm.br ^bb9(%426 : i64)
        ^bb9(%429: i64):  // 2 preds: ^bb8, ^bb10
          %430 = llvm.icmp "slt" %429, %427 : i64
          llvm.cond_br %430, ^bb10, ^bb11
        ^bb10:  // pred: ^bb9
          %431 = llvm.extractvalue %142[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %432 = llvm.mlir.constant(2 : index) : i64
          %433 = llvm.mul %424, %432  : i64
          %434 = llvm.add %433, %429  : i64
          %435 = llvm.getelementptr %431[%434] : (!llvm.ptr, i64) -> !llvm.ptr, f64
          %436 = llvm.load %435 : !llvm.ptr -> f64
          %437 = llvm.call @printf(%417, %436) : (!llvm.ptr<i8>, f64) -> i32
          %438 = llvm.add %429, %428  : i64
          llvm.br ^bb9(%438 : i64)
        ^bb11:  // pred: ^bb9
          %439 = llvm.call @printf(%420) : (!llvm.ptr<i8>) -> i32
          %440 = llvm.add %424, %423  : i64
          llvm.br ^bb7(%440 : i64)
        ^bb12:  // pred: ^bb7
          %441 = llvm.mlir.addressof @frmt_spec : !llvm.ptr<array<4 x i8>>
          %442 = llvm.mlir.constant(0 : index) : i64
          %443 = llvm.getelementptr %441[%442, %442] : (!llvm.ptr<array<4 x i8>>, i64, i64) -> !llvm.ptr<i8>
          %444 = llvm.mlir.addressof @nl : !llvm.ptr<array<2 x i8>>
          %445 = llvm.mlir.constant(0 : index) : i64
          %446 = llvm.getelementptr %444[%445, %445] : (!llvm.ptr<array<2 x i8>>, i64, i64) -> !llvm.ptr<i8>
          %447 = llvm.mlir.constant(0 : index) : i64
          %448 = llvm.mlir.constant(3 : index) : i64
          %449 = llvm.mlir.constant(1 : index) : i64
          llvm.br ^bb13(%447 : i64)
        ^bb13(%450: i64):  // 2 preds: ^bb12, ^bb17
          %451 = llvm.icmp "slt" %450, %448 : i64
          llvm.cond_br %451, ^bb14, ^bb18
        ^bb14:  // pred: ^bb13
          %452 = llvm.mlir.constant(0 : index) : i64
          %453 = llvm.mlir.constant(2 : index) : i64
          %454 = llvm.mlir.constant(1 : index) : i64
          llvm.br ^bb15(%452 : i64)
        ^bb15(%455: i64):  // 2 preds: ^bb14, ^bb16
          %456 = llvm.icmp "slt" %455, %453 : i64
          llvm.cond_br %456, ^bb16, ^bb17
        ^bb16:  // pred: ^bb15
          %457 = llvm.extractvalue %125[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %458 = llvm.mlir.constant(2 : index) : i64
          %459 = llvm.mul %450, %458  : i64
          %460 = llvm.add %459, %455  : i64
          %461 = llvm.getelementptr %457[%460] : (!llvm.ptr, i64) -> !llvm.ptr, f64
          %462 = llvm.load %461 : !llvm.ptr -> f64
          %463 = llvm.call @printf(%443, %462) : (!llvm.ptr<i8>, f64) -> i32
          %464 = llvm.add %455, %454  : i64
          llvm.br ^bb15(%464 : i64)
        ^bb17:  // pred: ^bb15
          %465 = llvm.call @printf(%446) : (!llvm.ptr<i8>) -> i32
          %466 = llvm.add %450, %449  : i64
          llvm.br ^bb13(%466 : i64)
        ^bb18:  // pred: ^bb13
          %467 = llvm.mlir.constant(0 : index) : i64
          %468 = llvm.mlir.constant(2 : index) : i64
          %469 = llvm.mlir.constant(1 : index) : i64
          llvm.br ^bb19(%467 : i64)
        ^bb19(%470: i64):  // 2 preds: ^bb18, ^bb23
          %471 = llvm.icmp "slt" %470, %468 : i64
          llvm.cond_br %471, ^bb20, ^bb24
        ^bb20:  // pred: ^bb19
          %472 = llvm.mlir.constant(0 : index) : i64
          %473 = llvm.mlir.constant(3 : index) : i64
          %474 = llvm.mlir.constant(1 : index) : i64
          llvm.br ^bb21(%472 : i64)
        ^bb21(%475: i64):  // 2 preds: ^bb20, ^bb22
          %476 = llvm.icmp "slt" %475, %473 : i64
          llvm.cond_br %476, ^bb22, ^bb23
        ^bb22:  // pred: ^bb21
          %477 = llvm.extractvalue %176[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %478 = llvm.mlir.constant(3 : index) : i64
          %479 = llvm.mul %470, %478  : i64
          %480 = llvm.add %479, %475  : i64
          %481 = llvm.getelementptr %477[%480] : (!llvm.ptr, i64) -> !llvm.ptr, i64
          %482 = llvm.load %481 : !llvm.ptr -> i64
          %483 = llvm.extractvalue %159[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %484 = llvm.mlir.constant(3 : index) : i64
          %485 = llvm.mul %470, %484  : i64
          %486 = llvm.add %485, %475  : i64
          %487 = llvm.getelementptr %483[%486] : (!llvm.ptr, i64) -> !llvm.ptr, i64
          %488 = llvm.load %487 : !llvm.ptr -> i64
          %489 = llvm.or %482, %488  : i64
          %490 = llvm.extractvalue %108[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %491 = llvm.mlir.constant(3 : index) : i64
          %492 = llvm.mul %470, %491  : i64
          %493 = llvm.add %492, %475  : i64
          %494 = llvm.getelementptr %490[%493] : (!llvm.ptr, i64) -> !llvm.ptr, i64
          llvm.store %489, %494 : i64, !llvm.ptr
          %495 = llvm.add %475, %474  : i64
          llvm.br ^bb21(%495 : i64)
        ^bb23:  // pred: ^bb21
          %496 = llvm.add %470, %469  : i64
          llvm.br ^bb19(%496 : i64)
        ^bb24:  // pred: ^bb19
          %497 = llvm.mlir.constant(0 : index) : i64
          %498 = llvm.mlir.constant(2 : index) : i64
          %499 = llvm.mlir.constant(1 : index) : i64
          llvm.br ^bb25(%497 : i64)
        ^bb25(%500: i64):  // 2 preds: ^bb24, ^bb29
          %501 = llvm.icmp "slt" %500, %498 : i64
          llvm.cond_br %501, ^bb26, ^bb30
        ^bb26:  // pred: ^bb25
          %502 = llvm.mlir.constant(0 : index) : i64
          %503 = llvm.mlir.constant(3 : index) : i64
          %504 = llvm.mlir.constant(1 : index) : i64
          llvm.br ^bb27(%502 : i64)
        ^bb27(%505: i64):  // 2 preds: ^bb26, ^bb28
          %506 = llvm.icmp "slt" %505, %503 : i64
          llvm.cond_br %506, ^bb28, ^bb29
        ^bb28:  // pred: ^bb27
          %507 = llvm.extractvalue %176[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %508 = llvm.mlir.constant(3 : index) : i64
          %509 = llvm.mul %500, %508  : i64
          %510 = llvm.add %509, %505  : i64
          %511 = llvm.getelementptr %507[%510] : (!llvm.ptr, i64) -> !llvm.ptr, i64
          %512 = llvm.load %511 : !llvm.ptr -> i64
          %513 = llvm.extractvalue %159[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %514 = llvm.mlir.constant(3 : index) : i64
          %515 = llvm.mul %500, %514  : i64
          %516 = llvm.add %515, %505  : i64
          %517 = llvm.getelementptr %513[%516] : (!llvm.ptr, i64) -> !llvm.ptr, i64
          %518 = llvm.load %517 : !llvm.ptr -> i64
          %519 = llvm.and %512, %518  : i64
          %520 = llvm.extractvalue %91[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %521 = llvm.mlir.constant(3 : index) : i64
          %522 = llvm.mul %500, %521  : i64
          %523 = llvm.add %522, %505  : i64
          %524 = llvm.getelementptr %520[%523] : (!llvm.ptr, i64) -> !llvm.ptr, i64
          llvm.store %519, %524 : i64, !llvm.ptr
          %525 = llvm.add %505, %504  : i64
          llvm.br ^bb27(%525 : i64)
        ^bb29:  // pred: ^bb27
          %526 = llvm.add %500, %499  : i64
          llvm.br ^bb25(%526 : i64)
        ^bb30:  // pred: ^bb25
          %527 = llvm.mlir.addressof @frmt_spec : !llvm.ptr<array<4 x i8>>
          %528 = llvm.mlir.constant(0 : index) : i64
          %529 = llvm.getelementptr %527[%528, %528] : (!llvm.ptr<array<4 x i8>>, i64, i64) -> !llvm.ptr<i8>
          %530 = llvm.mlir.addressof @nl : !llvm.ptr<array<2 x i8>>
          %531 = llvm.mlir.constant(0 : index) : i64
          %532 = llvm.getelementptr %530[%531, %531] : (!llvm.ptr<array<2 x i8>>, i64, i64) -> !llvm.ptr<i8>
          %533 = llvm.mlir.constant(0 : index) : i64
          %534 = llvm.mlir.constant(2 : index) : i64
          %535 = llvm.mlir.constant(1 : index) : i64
          llvm.br ^bb31(%533 : i64)
        ^bb31(%536: i64):  // 2 preds: ^bb30, ^bb35
          %537 = llvm.icmp "slt" %536, %534 : i64
          llvm.cond_br %537, ^bb32, ^bb36
        ^bb32:  // pred: ^bb31
          %538 = llvm.mlir.constant(0 : index) : i64
          %539 = llvm.mlir.constant(3 : index) : i64
          %540 = llvm.mlir.constant(1 : index) : i64
          llvm.br ^bb33(%538 : i64)
        ^bb33(%541: i64):  // 2 preds: ^bb32, ^bb34
          %542 = llvm.icmp "slt" %541, %539 : i64
          llvm.cond_br %542, ^bb34, ^bb35
        ^bb34:  // pred: ^bb33
          %543 = llvm.extractvalue %108[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %544 = llvm.mlir.constant(3 : index) : i64
          %545 = llvm.mul %536, %544  : i64
          %546 = llvm.add %545, %541  : i64
          %547 = llvm.getelementptr %543[%546] : (!llvm.ptr, i64) -> !llvm.ptr, i64
          %548 = llvm.load %547 : !llvm.ptr -> i64
          %549 = llvm.call @printf(%529, %548) : (!llvm.ptr<i8>, i64) -> i32
          %550 = llvm.add %541, %540  : i64
          llvm.br ^bb33(%550 : i64)
        ^bb35:  // pred: ^bb33
          %551 = llvm.call @printf(%532) : (!llvm.ptr<i8>) -> i32
          %552 = llvm.add %536, %535  : i64
          llvm.br ^bb31(%552 : i64)
        ^bb36:  // pred: ^bb31
          %553 = llvm.mlir.addressof @frmt_spec : !llvm.ptr<array<4 x i8>>
          %554 = llvm.mlir.constant(0 : index) : i64
          %555 = llvm.getelementptr %553[%554, %554] : (!llvm.ptr<array<4 x i8>>, i64, i64) -> !llvm.ptr<i8>
          %556 = llvm.mlir.addressof @nl : !llvm.ptr<array<2 x i8>>
          %557 = llvm.mlir.constant(0 : index) : i64
          %558 = llvm.getelementptr %556[%557, %557] : (!llvm.ptr<array<2 x i8>>, i64, i64) -> !llvm.ptr<i8>
          %559 = llvm.mlir.constant(0 : index) : i64
          %560 = llvm.mlir.constant(2 : index) : i64
          %561 = llvm.mlir.constant(1 : index) : i64
          llvm.br ^bb37(%559 : i64)
        ^bb37(%562: i64):  // 2 preds: ^bb36, ^bb41
          %563 = llvm.icmp "slt" %562, %560 : i64
          llvm.cond_br %563, ^bb38, ^bb42
        ^bb38:  // pred: ^bb37
          %564 = llvm.mlir.constant(0 : index) : i64
          %565 = llvm.mlir.constant(3 : index) : i64
          %566 = llvm.mlir.constant(1 : index) : i64
          llvm.br ^bb39(%564 : i64)
        ^bb39(%567: i64):  // 2 preds: ^bb38, ^bb40
          %568 = llvm.icmp "slt" %567, %565 : i64
          llvm.cond_br %568, ^bb40, ^bb41
        ^bb40:  // pred: ^bb39
          %569 = llvm.extractvalue %91[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %570 = llvm.mlir.constant(3 : index) : i64
          %571 = llvm.mul %562, %570  : i64
          %572 = llvm.add %571, %567  : i64
          %573 = llvm.getelementptr %569[%572] : (!llvm.ptr, i64) -> !llvm.ptr, i64
          %574 = llvm.load %573 : !llvm.ptr -> i64
          %575 = llvm.call @printf(%555, %574) : (!llvm.ptr<i8>, i64) -> i32
          %576 = llvm.add %567, %566  : i64
          llvm.br ^bb39(%576 : i64)
        ^bb41:  // pred: ^bb39
          %577 = llvm.call @printf(%558) : (!llvm.ptr<i8>) -> i32
          %578 = llvm.add %562, %561  : i64
          llvm.br ^bb37(%578 : i64)
        ^bb42:  // pred: ^bb37
          %579 = llvm.mlir.constant(0 : index) : i64
          %580 = llvm.mlir.constant(2 : index) : i64
          %581 = llvm.mlir.constant(1 : index) : i64
          llvm.br ^bb43(%579 : i64)
        ^bb43(%582: i64):  // 2 preds: ^bb42, ^bb47
          %583 = llvm.icmp "slt" %582, %580 : i64
          llvm.cond_br %583, ^bb44, ^bb48
        ^bb44:  // pred: ^bb43
          %584 = llvm.mlir.constant(0 : index) : i64
          %585 = llvm.mlir.constant(3 : index) : i64
          %586 = llvm.mlir.constant(1 : index) : i64
          llvm.br ^bb45(%584 : i64)
        ^bb45(%587: i64):  // 2 preds: ^bb44, ^bb46
          %588 = llvm.icmp "slt" %587, %585 : i64
          llvm.cond_br %588, ^bb46, ^bb47
        ^bb46:  // pred: ^bb45
          %589 = llvm.extractvalue %210[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %590 = llvm.mlir.constant(3 : index) : i64
          %591 = llvm.mul %582, %590  : i64
          %592 = llvm.add %591, %587  : i64
          %593 = llvm.getelementptr %589[%592] : (!llvm.ptr, i64) -> !llvm.ptr, f64
          %594 = llvm.load %593 : !llvm.ptr -> f64
          %595 = llvm.extractvalue %193[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %596 = llvm.mlir.constant(3 : index) : i64
          %597 = llvm.mul %582, %596  : i64
          %598 = llvm.add %597, %587  : i64
          %599 = llvm.getelementptr %595[%598] : (!llvm.ptr, i64) -> !llvm.ptr, f64
          %600 = llvm.load %599 : !llvm.ptr -> f64
          %601 = llvm.fadd %594, %600  : f64
          %602 = llvm.extractvalue %74[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %603 = llvm.mlir.constant(3 : index) : i64
          %604 = llvm.mul %582, %603  : i64
          %605 = llvm.add %604, %587  : i64
          %606 = llvm.getelementptr %602[%605] : (!llvm.ptr, i64) -> !llvm.ptr, f64
          llvm.store %601, %606 : f64, !llvm.ptr
          %607 = llvm.add %587, %586  : i64
          llvm.br ^bb45(%607 : i64)
        ^bb47:  // pred: ^bb45
          %608 = llvm.add %582, %581  : i64
          llvm.br ^bb43(%608 : i64)
        ^bb48:  // pred: ^bb43
          %609 = llvm.mlir.constant(0 : index) : i64
          %610 = llvm.mlir.constant(2 : index) : i64
          %611 = llvm.mlir.constant(1 : index) : i64
          llvm.br ^bb49(%609 : i64)
        ^bb49(%612: i64):  // 2 preds: ^bb48, ^bb53
          %613 = llvm.icmp "slt" %612, %610 : i64
          llvm.cond_br %613, ^bb50, ^bb54
        ^bb50:  // pred: ^bb49
          %614 = llvm.mlir.constant(0 : index) : i64
          %615 = llvm.mlir.constant(3 : index) : i64
          %616 = llvm.mlir.constant(1 : index) : i64
          llvm.br ^bb51(%614 : i64)
        ^bb51(%617: i64):  // 2 preds: ^bb50, ^bb52
          %618 = llvm.icmp "slt" %617, %615 : i64
          llvm.cond_br %618, ^bb52, ^bb53
        ^bb52:  // pred: ^bb51
          %619 = llvm.extractvalue %210[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %620 = llvm.mlir.constant(3 : index) : i64
          %621 = llvm.mul %612, %620  : i64
          %622 = llvm.add %621, %617  : i64
          %623 = llvm.getelementptr %619[%622] : (!llvm.ptr, i64) -> !llvm.ptr, f64
          %624 = llvm.load %623 : !llvm.ptr -> f64
          %625 = llvm.extractvalue %193[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %626 = llvm.mlir.constant(3 : index) : i64
          %627 = llvm.mul %612, %626  : i64
          %628 = llvm.add %627, %617  : i64
          %629 = llvm.getelementptr %625[%628] : (!llvm.ptr, i64) -> !llvm.ptr, f64
          %630 = llvm.load %629 : !llvm.ptr -> f64
          %631 = llvm.fsub %624, %630  : f64
          %632 = llvm.extractvalue %57[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %633 = llvm.mlir.constant(3 : index) : i64
          %634 = llvm.mul %612, %633  : i64
          %635 = llvm.add %634, %617  : i64
          %636 = llvm.getelementptr %632[%635] : (!llvm.ptr, i64) -> !llvm.ptr, f64
          llvm.store %631, %636 : f64, !llvm.ptr
          %637 = llvm.add %617, %616  : i64
          llvm.br ^bb51(%637 : i64)
        ^bb53:  // pred: ^bb51
          %638 = llvm.add %612, %611  : i64
          llvm.br ^bb49(%638 : i64)
        ^bb54:  // pred: ^bb49
          %639 = llvm.mlir.constant(0 : index) : i64
          %640 = llvm.mlir.constant(2 : index) : i64
          %641 = llvm.mlir.constant(1 : index) : i64
          llvm.br ^bb55(%639 : i64)
        ^bb55(%642: i64):  // 2 preds: ^bb54, ^bb59
          %643 = llvm.icmp "slt" %642, %640 : i64
          llvm.cond_br %643, ^bb56, ^bb60
        ^bb56:  // pred: ^bb55
          %644 = llvm.mlir.constant(0 : index) : i64
          %645 = llvm.mlir.constant(3 : index) : i64
          %646 = llvm.mlir.constant(1 : index) : i64
          llvm.br ^bb57(%644 : i64)
        ^bb57(%647: i64):  // 2 preds: ^bb56, ^bb58
          %648 = llvm.icmp "slt" %647, %645 : i64
          llvm.cond_br %648, ^bb58, ^bb59
        ^bb58:  // pred: ^bb57
          %649 = llvm.extractvalue %210[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %650 = llvm.mlir.constant(3 : index) : i64
          %651 = llvm.mul %642, %650  : i64
          %652 = llvm.add %651, %647  : i64
          %653 = llvm.getelementptr %649[%652] : (!llvm.ptr, i64) -> !llvm.ptr, f64
          %654 = llvm.load %653 : !llvm.ptr -> f64
          %655 = llvm.extractvalue %193[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %656 = llvm.mlir.constant(3 : index) : i64
          %657 = llvm.mul %642, %656  : i64
          %658 = llvm.add %657, %647  : i64
          %659 = llvm.getelementptr %655[%658] : (!llvm.ptr, i64) -> !llvm.ptr, f64
          %660 = llvm.load %659 : !llvm.ptr -> f64
          %661 = llvm.fmul %654, %660  : f64
          %662 = llvm.extractvalue %40[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %663 = llvm.mlir.constant(3 : index) : i64
          %664 = llvm.mul %642, %663  : i64
          %665 = llvm.add %664, %647  : i64
          %666 = llvm.getelementptr %662[%665] : (!llvm.ptr, i64) -> !llvm.ptr, f64
          llvm.store %661, %666 : f64, !llvm.ptr
          %667 = llvm.add %647, %646  : i64
          llvm.br ^bb57(%667 : i64)
        ^bb59:  // pred: ^bb57
          %668 = llvm.add %642, %641  : i64
          llvm.br ^bb55(%668 : i64)
        ^bb60:  // pred: ^bb55
          %669 = llvm.mlir.addressof @frmt_spec : !llvm.ptr<array<4 x i8>>
          %670 = llvm.mlir.constant(0 : index) : i64
          %671 = llvm.getelementptr %669[%670, %670] : (!llvm.ptr<array<4 x i8>>, i64, i64) -> !llvm.ptr<i8>
          %672 = llvm.mlir.addressof @nl : !llvm.ptr<array<2 x i8>>
          %673 = llvm.mlir.constant(0 : index) : i64
          %674 = llvm.getelementptr %672[%673, %673] : (!llvm.ptr<array<2 x i8>>, i64, i64) -> !llvm.ptr<i8>
          %675 = llvm.mlir.constant(0 : index) : i64
          %676 = llvm.mlir.constant(2 : index) : i64
          %677 = llvm.mlir.constant(1 : index) : i64
          llvm.br ^bb61(%675 : i64)
        ^bb61(%678: i64):  // 2 preds: ^bb60, ^bb65
          %679 = llvm.icmp "slt" %678, %676 : i64
          llvm.cond_br %679, ^bb62, ^bb66
        ^bb62:  // pred: ^bb61
          %680 = llvm.mlir.constant(0 : index) : i64
          %681 = llvm.mlir.constant(3 : index) : i64
          %682 = llvm.mlir.constant(1 : index) : i64
          llvm.br ^bb63(%680 : i64)
        ^bb63(%683: i64):  // 2 preds: ^bb62, ^bb64
          %684 = llvm.icmp "slt" %683, %681 : i64
          llvm.cond_br %684, ^bb64, ^bb65
        ^bb64:  // pred: ^bb63
          %685 = llvm.extractvalue %74[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %686 = llvm.mlir.constant(3 : index) : i64
          %687 = llvm.mul %678, %686  : i64
          %688 = llvm.add %687, %683  : i64
          %689 = llvm.getelementptr %685[%688] : (!llvm.ptr, i64) -> !llvm.ptr, f64
          %690 = llvm.load %689 : !llvm.ptr -> f64
          %691 = llvm.call @printf(%671, %690) : (!llvm.ptr<i8>, f64) -> i32
          %692 = llvm.add %683, %682  : i64
          llvm.br ^bb63(%692 : i64)
        ^bb65:  // pred: ^bb63
          %693 = llvm.call @printf(%674) : (!llvm.ptr<i8>) -> i32
          %694 = llvm.add %678, %677  : i64
          llvm.br ^bb61(%694 : i64)
        ^bb66:  // pred: ^bb61
          %695 = llvm.mlir.addressof @frmt_spec : !llvm.ptr<array<4 x i8>>
          %696 = llvm.mlir.constant(0 : index) : i64
          %697 = llvm.getelementptr %695[%696, %696] : (!llvm.ptr<array<4 x i8>>, i64, i64) -> !llvm.ptr<i8>
          %698 = llvm.mlir.addressof @nl : !llvm.ptr<array<2 x i8>>
          %699 = llvm.mlir.constant(0 : index) : i64
          %700 = llvm.getelementptr %698[%699, %699] : (!llvm.ptr<array<2 x i8>>, i64, i64) -> !llvm.ptr<i8>
          %701 = llvm.mlir.constant(0 : index) : i64
          %702 = llvm.mlir.constant(2 : index) : i64
          %703 = llvm.mlir.constant(1 : index) : i64
          llvm.br ^bb67(%701 : i64)
        ^bb67(%704: i64):  // 2 preds: ^bb66, ^bb71
          %705 = llvm.icmp "slt" %704, %702 : i64
          llvm.cond_br %705, ^bb68, ^bb72
        ^bb68:  // pred: ^bb67
          %706 = llvm.mlir.constant(0 : index) : i64
          %707 = llvm.mlir.constant(3 : index) : i64
          %708 = llvm.mlir.constant(1 : index) : i64
          llvm.br ^bb69(%706 : i64)
        ^bb69(%709: i64):  // 2 preds: ^bb68, ^bb70
          %710 = llvm.icmp "slt" %709, %707 : i64
          llvm.cond_br %710, ^bb70, ^bb71
        ^bb70:  // pred: ^bb69
          %711 = llvm.extractvalue %57[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %712 = llvm.mlir.constant(3 : index) : i64
          %713 = llvm.mul %704, %712  : i64
          %714 = llvm.add %713, %709  : i64
          %715 = llvm.getelementptr %711[%714] : (!llvm.ptr, i64) -> !llvm.ptr, f64
          %716 = llvm.load %715 : !llvm.ptr -> f64
          %717 = llvm.call @printf(%697, %716) : (!llvm.ptr<i8>, f64) -> i32
          %718 = llvm.add %709, %708  : i64
          llvm.br ^bb69(%718 : i64)
        ^bb71:  // pred: ^bb69
          %719 = llvm.call @printf(%700) : (!llvm.ptr<i8>) -> i32
          %720 = llvm.add %704, %703  : i64
          llvm.br ^bb67(%720 : i64)
        ^bb72:  // pred: ^bb67
          %721 = llvm.mlir.addressof @frmt_spec : !llvm.ptr<array<4 x i8>>
          %722 = llvm.mlir.constant(0 : index) : i64
          %723 = llvm.getelementptr %721[%722, %722] : (!llvm.ptr<array<4 x i8>>, i64, i64) -> !llvm.ptr<i8>
          %724 = llvm.mlir.addressof @nl : !llvm.ptr<array<2 x i8>>
          %725 = llvm.mlir.constant(0 : index) : i64
          %726 = llvm.getelementptr %724[%725, %725] : (!llvm.ptr<array<2 x i8>>, i64, i64) -> !llvm.ptr<i8>
          %727 = llvm.mlir.constant(0 : index) : i64
          %728 = llvm.mlir.constant(2 : index) : i64
          %729 = llvm.mlir.constant(1 : index) : i64
          llvm.br ^bb73(%727 : i64)
        ^bb73(%730: i64):  // 2 preds: ^bb72, ^bb77
          %731 = llvm.icmp "slt" %730, %728 : i64
          llvm.cond_br %731, ^bb74, ^bb78
        ^bb74:  // pred: ^bb73
          %732 = llvm.mlir.constant(0 : index) : i64
          %733 = llvm.mlir.constant(3 : index) : i64
          %734 = llvm.mlir.constant(1 : index) : i64
          llvm.br ^bb75(%732 : i64)
        ^bb75(%735: i64):  // 2 preds: ^bb74, ^bb76
          %736 = llvm.icmp "slt" %735, %733 : i64
          llvm.cond_br %736, ^bb76, ^bb77
        ^bb76:  // pred: ^bb75
          %737 = llvm.extractvalue %40[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          %738 = llvm.mlir.constant(3 : index) : i64
          %739 = llvm.mul %730, %738  : i64
          %740 = llvm.add %739, %735  : i64
          %741 = llvm.getelementptr %737[%740] : (!llvm.ptr, i64) -> !llvm.ptr, f64
          %742 = llvm.load %741 : !llvm.ptr -> f64
          %743 = llvm.call @printf(%723, %742) : (!llvm.ptr<i8>, f64) -> i32
          %744 = llvm.add %735, %734  : i64
          llvm.br ^bb75(%744 : i64)
        ^bb77:  // pred: ^bb75
          %745 = llvm.call @printf(%726) : (!llvm.ptr<i8>) -> i32
          %746 = llvm.add %730, %729  : i64
          llvm.br ^bb73(%746 : i64)
        ^bb78:  // pred: ^bb73
          %747 = llvm.extractvalue %210[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          llvm.call @free(%747) : (!llvm.ptr) -> ()
          %748 = llvm.extractvalue %193[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          llvm.call @free(%748) : (!llvm.ptr) -> ()
          %749 = llvm.extractvalue %176[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          llvm.call @free(%749) : (!llvm.ptr) -> ()
          %750 = llvm.extractvalue %159[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          llvm.call @free(%750) : (!llvm.ptr) -> ()
          %751 = llvm.extractvalue %142[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          llvm.call @free(%751) : (!llvm.ptr) -> ()
          %752 = llvm.extractvalue %125[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          llvm.call @free(%752) : (!llvm.ptr) -> ()
          %753 = llvm.extractvalue %108[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          llvm.call @free(%753) : (!llvm.ptr) -> ()
          %754 = llvm.extractvalue %91[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          llvm.call @free(%754) : (!llvm.ptr) -> ()
          %755 = llvm.extractvalue %74[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          llvm.call @free(%755) : (!llvm.ptr) -> ()
          %756 = llvm.extractvalue %57[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          llvm.call @free(%756) : (!llvm.ptr) -> ()
          %757 = llvm.extractvalue %40[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
          llvm.call @free(%757) : (!llvm.ptr) -> ()
          llvm.return
        }
      }
    ```

2. 执行`./build/bin/toyc-ch6 -emit=llvm ./toy_mod/src/Ch6/codegen.toy -opt`命令就可以生成llvm IR

    ```llvm
      ; ModuleID = 'LLVMDialectModule'
      source_filename = "LLVMDialectModule"
      target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
      target triple = "x86_64-unknown-linux-gnu"

      @frmt_spec = internal constant [4 x i8] c"%f \00"

      ; Function Attrs: mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite)
      declare noalias noundef ptr @malloc(i64 noundef) local_unnamed_addr #0

      ; Function Attrs: mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite)
      declare void @free(ptr allocptr nocapture noundef) local_unnamed_addr #1

      ; Function Attrs: nofree nounwind
      declare noundef i32 @printf(ptr nocapture noundef readonly, ...) local_unnamed_addr #2

      ; Function Attrs: nounwind
      define void @main() local_unnamed_addr #3 !dbg !3 {
      .preheader42:
        %0 = tail call dereferenceable_or_null(48) ptr @malloc(i64 48), !dbg !6
        %1 = tail call dereferenceable_or_null(48) ptr @malloc(i64 48), !dbg !7
        store double 1.100000e+00, ptr %1, align 8, !dbg !7
        %2 = getelementptr double, ptr %1, i64 1, !dbg !7
        store double 2.200000e+00, ptr %2, align 8, !dbg !7
        %3 = getelementptr double, ptr %1, i64 2, !dbg !7
        store double 3.300000e+00, ptr %3, align 8, !dbg !7
        %4 = getelementptr double, ptr %1, i64 3, !dbg !7
        store double 4.400000e+00, ptr %4, align 8, !dbg !7
        %5 = getelementptr double, ptr %1, i64 4, !dbg !7
        store double 5.500000e+00, ptr %5, align 8, !dbg !7
        %6 = getelementptr double, ptr %1, i64 5, !dbg !7
        store double 6.600000e+00, ptr %6, align 8, !dbg !7
        store double 1.000000e+00, ptr %0, align 8, !dbg !6
        %7 = getelementptr double, ptr %0, i64 1, !dbg !6
        store double 2.000000e+00, ptr %7, align 8, !dbg !6
        %8 = getelementptr double, ptr %0, i64 2, !dbg !6
        store double 3.000000e+00, ptr %8, align 8, !dbg !6
        %9 = getelementptr double, ptr %0, i64 3, !dbg !6
        store double 4.000000e+00, ptr %9, align 8, !dbg !6
        %10 = getelementptr double, ptr %0, i64 4, !dbg !6
        store double 5.000000e+00, ptr %10, align 8, !dbg !6
        %11 = getelementptr double, ptr %0, i64 5, !dbg !6
        store double 6.000000e+00, ptr %11, align 8, !dbg !6
        %12 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double 1.100000e+00), !dbg !8
        %13 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double 1.760000e+01), !dbg !8
        %putchar18 = tail call i32 @putchar(i32 10), !dbg !8
        %14 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double 4.400000e+00), !dbg !8
        %15 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double 2.750000e+01), !dbg !8
        %putchar18.1 = tail call i32 @putchar(i32 10), !dbg !8
        %16 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double 0x4023CCCCCCCCCCCC), !dbg !8
        %17 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double 0x4043CCCCCCCCCCCC), !dbg !8
        %putchar18.2 = tail call i32 @putchar(i32 10), !dbg !8
        %18 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double 1.100000e+00), !dbg !9
        %19 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double 1.760000e+01), !dbg !9
        %putchar17 = tail call i32 @putchar(i32 10), !dbg !9
        %20 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double 4.400000e+00), !dbg !9
        %21 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double 2.750000e+01), !dbg !9
        %putchar17.1 = tail call i32 @putchar(i32 10), !dbg !9
        %22 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double 0x4023CCCCCCCCCCCC), !dbg !9
        %23 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double 0x4043CCCCCCCCCCCC), !dbg !9
        %putchar17.2 = tail call i32 @putchar(i32 10), !dbg !9
        %24 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, i64 15), !dbg !10
        %25 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, i64 14), !dbg !10
        %26 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, i64 15), !dbg !10
        %putchar16 = tail call i32 @putchar(i32 10), !dbg !10
        %27 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, i64 26), !dbg !10
        %28 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, i64 27), !dbg !10
        %29 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, i64 30), !dbg !10
        %putchar16.1 = tail call i32 @putchar(i32 10), !dbg !10
        %30 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, i64 5), !dbg !11
        %31 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, i64 8), !dbg !11
        %32 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, i64 9), !dbg !11
        %putchar15 = tail call i32 @putchar(i32 10), !dbg !11
        %33 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, i64 0), !dbg !11
        %34 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, i64 1), !dbg !11
        %35 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, i64 0), !dbg !11
        %putchar15.1 = tail call i32 @putchar(i32 10), !dbg !11
        %36 = load double, ptr %1, align 8, !dbg !12
        %37 = load double, ptr %0, align 8, !dbg !12
        %38 = fadd double %36, %37, !dbg !12
        %39 = load double, ptr %2, align 8, !dbg !12
        %40 = load double, ptr %7, align 8, !dbg !12
        %41 = fadd double %39, %40, !dbg !12
        %42 = load double, ptr %3, align 8, !dbg !12
        %43 = load double, ptr %8, align 8, !dbg !12
        %44 = fadd double %42, %43, !dbg !12
        %45 = load double, ptr %4, align 8, !dbg !12
        %46 = load double, ptr %9, align 8, !dbg !12
        %47 = fadd double %45, %46, !dbg !12
        %48 = load double, ptr %5, align 8, !dbg !12
        %49 = load double, ptr %10, align 8, !dbg !12
        %50 = fadd double %48, %49, !dbg !12
        %51 = load double, ptr %6, align 8, !dbg !12
        %52 = load double, ptr %11, align 8, !dbg !12
        %53 = fadd double %51, %52, !dbg !12
        %54 = fsub double %36, %37, !dbg !13
        %55 = fsub double %39, %40, !dbg !13
        %56 = fsub double %42, %43, !dbg !13
        %57 = fsub double %45, %46, !dbg !13
        %58 = fsub double %48, %49, !dbg !13
        %59 = fsub double %51, %52, !dbg !13
        %60 = fmul double %36, %37, !dbg !14
        %61 = fmul double %39, %40, !dbg !14
        %62 = fmul double %42, %43, !dbg !14
        %63 = fmul double %45, %46, !dbg !14
        %64 = fmul double %48, %49, !dbg !14
        %65 = fmul double %51, %52, !dbg !14
        %66 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double %38), !dbg !15
        %67 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double %41), !dbg !15
        %68 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double %44), !dbg !15
        %putchar14 = tail call i32 @putchar(i32 10), !dbg !15
        %69 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double %47), !dbg !15
        %70 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double %50), !dbg !15
        %71 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double %53), !dbg !15
        %putchar14.1 = tail call i32 @putchar(i32 10), !dbg !15
        %72 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double %54), !dbg !16
        %73 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double %55), !dbg !16
        %74 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double %56), !dbg !16
        %putchar13 = tail call i32 @putchar(i32 10), !dbg !16
        %75 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double %57), !dbg !16
        %76 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double %58), !dbg !16
        %77 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double %59), !dbg !16
        %putchar13.1 = tail call i32 @putchar(i32 10), !dbg !16
        %78 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double %60), !dbg !17
        %79 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double %61), !dbg !17
        %80 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double %62), !dbg !17
        %putchar = tail call i32 @putchar(i32 10), !dbg !17
        %81 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double %63), !dbg !17
        %82 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double %64), !dbg !17
        %83 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double %65), !dbg !17
        %putchar.1 = tail call i32 @putchar(i32 10), !dbg !17
        tail call void @free(ptr %1), !dbg !7
        tail call void @free(ptr %0), !dbg !6
        ret void, !dbg !18
      }

      ; Function Attrs: nofree nounwind
      declare noundef i32 @putchar(i32 noundef) local_unnamed_addr #2

      attributes #0 = { mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite) "alloc-family"="malloc" }
      attributes #1 = { mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite) "alloc-family"="malloc" }
      attributes #2 = { nofree nounwind }
      attributes #3 = { nounwind }

      !llvm.module.flags = !{!0}
      !llvm.dbg.cu = !{!1}

      !0 = !{i32 2, !"Debug Info Version", i32 3}
      !1 = distinct !DICompileUnit(language: DW_LANG_C, file: !2, producer: "MLIR", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
      !2 = !DIFile(filename: "codegen.toy", directory: "./toy_mod/src/Ch6")
      !3 = distinct !DISubprogram(name: "main", linkageName: "main", scope: !2, file: !2, line: 8, type: !4, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !1)
      !4 = !DISubroutineType(cc: DW_CC_normal, types: !5)
      !5 = !{}
      !6 = !DILocation(line: 11, scope: !3)
      !7 = !DILocation(line: 9, scope: !3)
      !8 = !DILocation(line: 20, column: 3, scope: !3)
      !9 = !DILocation(line: 21, column: 3, scope: !3)
      !10 = !DILocation(line: 26, column: 3, scope: !3)
      !11 = !DILocation(line: 27, column: 3, scope: !3)
      !12 = !DILocation(line: 29, column: 15, scope: !3)
      !13 = !DILocation(line: 30, column: 15, scope: !3)
      !14 = !DILocation(line: 31, column: 15, scope: !3)
      !15 = !DILocation(line: 32, column: 3, scope: !3)
      !16 = !DILocation(line: 33, column: 3, scope: !3)
      !17 = !DILocation(line: 34, column: 3, scope: !3)
      !18 = !DILocation(line: 8, column: 1, scope: !3)
    ```

3. 尝试使用`./build/bin/toyc-ch6 -emit=jit ./toy_mod/src/Ch6/codegen.toy -opt`命令运行代码会发现，所有的整数print的结果均为0

    ```
    >> ./build/bin/toyc-ch6 -emit=jit ./toy_mod/src/Ch6/codegen.toy -opt
    1.100000 17.600000 
    4.400000 27.500000 
    9.900000 39.600000 
    1.100000 17.600000 
    4.400000 27.500000 
    9.900000 39.600000 
    0.000000 0.000000 0.000000 
    0.000000 0.000000 0.000000 
    0.000000 0.000000 0.000000 
    0.000000 0.000000 0.000000 
    2.100000 4.200000 6.300000 
    8.400000 10.500000 12.600000 
    0.100000 0.200000 0.300000 
    0.400000 0.500000 0.600000 
    1.100000 4.400000 9.900000 
    17.600000 27.500000 39.600000 
    ```

4. 检查PrintOpLowering的代码会发现，Toy的Example在将PrintOP lowering到LLVM的printf时，格式标识符固定传了%f。
  
    ```c++
    class PrintOpLowering : public ConversionPattern {
      
      matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const override {
        ...

        Value formatSpecifierCst = getOrCreateGlobalString(
            loc, rewriter, "frmt_spec", StringRef("%f \0", 4), parentModule);
        Value newLineCst = getOrCreateGlobalString(
            loc, rewriter, "nl", StringRef("\n\0", 2), parentModule);

        ...

        // Generate a call to printf for the current element of the loop.
        auto printOp = cast<toy::PrintOp>(op);
        auto elementLoad =
            rewriter.create<memref::LoadOp>(loc, printOp.getInput(), loopIvs);
        rewriter.create<func::CallOp>(
            loc, printfRef, rewriter.getIntegerType(32),
            ArrayRef<Value>({formatSpecifierCst, elementLoad}));

        // Notify the rewriter that this operation has been removed.
        rewriter.eraseOp(op);
        return success();
      }

      ...

    }
    ```

5. 为了方便修改和debug，首先修改toyc.cpp的loadMLIR函数，使用`mlir::registerAllDialects()`注册所有dialects，使其能够直接读取已经lowering到llvm dialect的mlir

    ```c++
    // <toyc.cpp>

    ...

    class DialectRegistry;

    int loadMLIR(mlir::MLIRContext &context,
                mlir::OwningOpRef<mlir::ModuleOp> &module) {
      mlir::registerAllDialects(context);

      ...

    }
    ```

6. 修改PrintOpLowering, 使其能够判断print数据类型，使用相应的format speicifier。
  
    ```c++
    class PrintOpLowering : public ConversionPattern {
      
      matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const override {
        ...

        // 1. 分别定义integer和float的format specifier
        Value intFormatSpecifierCst = getOrCreateGlobalString(
            loc, rewriter, "int_frmt_spec", StringRef("%d \0", 4), parentModule);
        Value floatFormatSpecifierCst = getOrCreateGlobalString(
            loc, rewriter, "float_frmt_spec", StringRef("%f \0", 4), parentModule);
        Value newLineCst = getOrCreateGlobalString(
            loc, rewriter, "nl", StringRef("\n\0", 2), parentModule);

        ...

        // Generate a call to printf for the current element of the loop.
        auto printOp = cast<toy::PrintOp>(op);
        Value printOpInput = printOp.getInput();
        Type printOpInputType = printOpInput.getType();

        // 2. 获取PrintOP的Input类型
        auto printOpmemRefType = dyn_cast<MemRefType>(printOpInputType);
        Type elementType = printOpmemRefType.getElementType();

        if (::llvm::DebugFlag) {
          printOpInput.print(llvm::dbgs() << "\n");
          printOpInputType.print(llvm::dbgs() << "\n");
          elementType.print(llvm::dbgs() << "\n" );
        }

        auto elementLoad =
            rewriter.create<memref::LoadOp>(loc, printOp.getInput(), loopIvs);
        // 3. 对于浮点数，使用%f
        if (elementType.isF64()) {
          if (::llvm::DebugFlag)
            printf("print Float\n");
          rewriter.create<func::CallOp>(
            loc, printfRef, rewriter.getIntegerType(32),
            ArrayRef<Value>({floatFormatSpecifierCst, elementLoad}));
        } else {
        // 4. 对于整数，使用%d
          if (::llvm::DebugFlag)
            printf("print Integer\n");
          rewriter.create<func::CallOp>(
            loc, printfRef, rewriter.getIntegerType(32),
            ArrayRef<Value>({formatSpecifierCst, elementLoad}));
            ArrayRef<Value>({intFormatSpecifierCst, elementLoad}));
        }

      ...

    }
    ```

7. 重新编译后，再次执行`./build/bin/toyc-ch6 -emit=mlir-llvm ./toy_mod/src/Ch6/codegen.toy -opt`可以看到，整数和浮点数的printf分别调用了不同的global constant。

    ```mlir
    module {
      llvm.func @free(!llvm.ptr)
      llvm.mlir.global internal constant @nl("\0A\00") {addr_space = 0 : i32}
      llvm.mlir.global internal constant @float_frmt_spec("%f \00") {addr_space = 0 : i32}
      llvm.mlir.global internal constant @int_frmt_spec("%d \00") {addr_space = 0 : i32}
      llvm.func @printf(!llvm.ptr<i8>, ...) -> i32
      llvm.func @malloc(i64) -> !llvm.ptr
      llvm.func @main() {
        %0 = llvm.mlir.constant(18 : i64) : i64
        %1 = llvm.mlir.constant(17 : i64) : i64
        %2 = llvm.mlir.constant(16 : i64) : i64
        %3 = llvm.mlir.constant(15 : i64) : i64
        %4 = llvm.mlir.constant(14 : i64) : i64
        %5 = llvm.mlir.constant(13 : i64) : i64
        %6 = llvm.mlir.constant(12 : i64) : i64
        %7 = llvm.mlir.constant(11 : i64) : i64
        %8 = llvm.mlir.constant(10 : i64) : i64
        %9 = llvm.mlir.constant(9 : i64) : i64
        %10 = llvm.mlir.constant(8 : i64) : i64
        %11 = llvm.mlir.constant(7 : i64) : i64
        %12 = llvm.mlir.constant(6.000000e+00 : f64) : f64
        %13 = llvm.mlir.constant(5.000000e+00 : f64) : f64
        %14 = llvm.mlir.constant(4.000000e+00 : f64) : f64
        %15 = llvm.mlir.constant(3.000000e+00 : f64) : f64
        %16 = llvm.mlir.constant(2.000000e+00 : f64) : f64
        %17 = llvm.mlir.constant(1.000000e+00 : f64) : f64
        %18 = llvm.mlir.constant(6.600000e+00 : f64) : f64
        %19 = llvm.mlir.constant(5.500000e+00 : f64) : f64
        %20 = llvm.mlir.constant(4.400000e+00 : f64) : f64
        %21 = llvm.mlir.constant(3.300000e+00 : f64) : f64
        %22 = llvm.mlir.constant(2.200000e+00 : f64) : f64
        %23 = llvm.mlir.constant(1.100000e+00 : f64) : f64
        %24 = llvm.mlir.constant(2 : index) : i64
        %25 = llvm.mlir.constant(3 : index) : i64
        %26 = llvm.mlir.constant(1 : index) : i64
        %27 = llvm.mlir.constant(6 : index) : i64
        %28 = llvm.mlir.null : !llvm.ptr
        %29 = llvm.getelementptr %28[%27] : (!llvm.ptr, i64) -> !llvm.ptr, f64
        %30 = llvm.ptrtoint %29 : !llvm.ptr to i64
        %31 = llvm.call @malloc(%30) : (i64) -> !llvm.ptr
        %32 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
        %33 = llvm.insertvalue %31, %32[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %34 = llvm.insertvalue %31, %33[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %35 = llvm.mlir.constant(0 : index) : i64
        %36 = llvm.insertvalue %35, %34[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %37 = llvm.insertvalue %24, %36[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %38 = llvm.insertvalue %25, %37[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %39 = llvm.insertvalue %25, %38[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %40 = llvm.insertvalue %26, %39[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %41 = llvm.mlir.constant(2 : index) : i64
        %42 = llvm.mlir.constant(3 : index) : i64
        %43 = llvm.mlir.constant(1 : index) : i64
        %44 = llvm.mlir.constant(6 : index) : i64
        %45 = llvm.mlir.null : !llvm.ptr
        %46 = llvm.getelementptr %45[%44] : (!llvm.ptr, i64) -> !llvm.ptr, f64
        %47 = llvm.ptrtoint %46 : !llvm.ptr to i64
        %48 = llvm.call @malloc(%47) : (i64) -> !llvm.ptr
        %49 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
        %50 = llvm.insertvalue %48, %49[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %51 = llvm.insertvalue %48, %50[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %52 = llvm.mlir.constant(0 : index) : i64
        %53 = llvm.insertvalue %52, %51[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %54 = llvm.insertvalue %41, %53[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %55 = llvm.insertvalue %42, %54[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %56 = llvm.insertvalue %42, %55[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %57 = llvm.insertvalue %43, %56[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %58 = llvm.mlir.constant(2 : index) : i64
        %59 = llvm.mlir.constant(3 : index) : i64
        %60 = llvm.mlir.constant(1 : index) : i64
        %61 = llvm.mlir.constant(6 : index) : i64
        %62 = llvm.mlir.null : !llvm.ptr
        %63 = llvm.getelementptr %62[%61] : (!llvm.ptr, i64) -> !llvm.ptr, f64
        %64 = llvm.ptrtoint %63 : !llvm.ptr to i64
        %65 = llvm.call @malloc(%64) : (i64) -> !llvm.ptr
        %66 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
        %67 = llvm.insertvalue %65, %66[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %68 = llvm.insertvalue %65, %67[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %69 = llvm.mlir.constant(0 : index) : i64
        %70 = llvm.insertvalue %69, %68[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %71 = llvm.insertvalue %58, %70[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %72 = llvm.insertvalue %59, %71[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %73 = llvm.insertvalue %59, %72[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %74 = llvm.insertvalue %60, %73[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %75 = llvm.mlir.constant(2 : index) : i64
        %76 = llvm.mlir.constant(3 : index) : i64
        %77 = llvm.mlir.constant(1 : index) : i64
        %78 = llvm.mlir.constant(6 : index) : i64
        %79 = llvm.mlir.null : !llvm.ptr
        %80 = llvm.getelementptr %79[%78] : (!llvm.ptr, i64) -> !llvm.ptr, i64
        %81 = llvm.ptrtoint %80 : !llvm.ptr to i64
        %82 = llvm.call @malloc(%81) : (i64) -> !llvm.ptr
        %83 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
        %84 = llvm.insertvalue %82, %83[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %85 = llvm.insertvalue %82, %84[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %86 = llvm.mlir.constant(0 : index) : i64
        %87 = llvm.insertvalue %86, %85[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %88 = llvm.insertvalue %75, %87[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %89 = llvm.insertvalue %76, %88[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %90 = llvm.insertvalue %76, %89[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %91 = llvm.insertvalue %77, %90[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %92 = llvm.mlir.constant(2 : index) : i64
        %93 = llvm.mlir.constant(3 : index) : i64
        %94 = llvm.mlir.constant(1 : index) : i64
        %95 = llvm.mlir.constant(6 : index) : i64
        %96 = llvm.mlir.null : !llvm.ptr
        %97 = llvm.getelementptr %96[%95] : (!llvm.ptr, i64) -> !llvm.ptr, i64
        %98 = llvm.ptrtoint %97 : !llvm.ptr to i64
        %99 = llvm.call @malloc(%98) : (i64) -> !llvm.ptr
        %100 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
        %101 = llvm.insertvalue %99, %100[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %102 = llvm.insertvalue %99, %101[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %103 = llvm.mlir.constant(0 : index) : i64
        %104 = llvm.insertvalue %103, %102[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %105 = llvm.insertvalue %92, %104[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %106 = llvm.insertvalue %93, %105[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %107 = llvm.insertvalue %93, %106[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %108 = llvm.insertvalue %94, %107[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %109 = llvm.mlir.constant(3 : index) : i64
        %110 = llvm.mlir.constant(2 : index) : i64
        %111 = llvm.mlir.constant(1 : index) : i64
        %112 = llvm.mlir.constant(6 : index) : i64
        %113 = llvm.mlir.null : !llvm.ptr
        %114 = llvm.getelementptr %113[%112] : (!llvm.ptr, i64) -> !llvm.ptr, f64
        %115 = llvm.ptrtoint %114 : !llvm.ptr to i64
        %116 = llvm.call @malloc(%115) : (i64) -> !llvm.ptr
        %117 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
        %118 = llvm.insertvalue %116, %117[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %119 = llvm.insertvalue %116, %118[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %120 = llvm.mlir.constant(0 : index) : i64
        %121 = llvm.insertvalue %120, %119[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %122 = llvm.insertvalue %109, %121[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %123 = llvm.insertvalue %110, %122[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %124 = llvm.insertvalue %110, %123[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %125 = llvm.insertvalue %111, %124[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %126 = llvm.mlir.constant(3 : index) : i64
        %127 = llvm.mlir.constant(2 : index) : i64
        %128 = llvm.mlir.constant(1 : index) : i64
        %129 = llvm.mlir.constant(6 : index) : i64
        %130 = llvm.mlir.null : !llvm.ptr
        %131 = llvm.getelementptr %130[%129] : (!llvm.ptr, i64) -> !llvm.ptr, f64
        %132 = llvm.ptrtoint %131 : !llvm.ptr to i64
        %133 = llvm.call @malloc(%132) : (i64) -> !llvm.ptr
        %134 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
        %135 = llvm.insertvalue %133, %134[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %136 = llvm.insertvalue %133, %135[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %137 = llvm.mlir.constant(0 : index) : i64
        %138 = llvm.insertvalue %137, %136[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %139 = llvm.insertvalue %126, %138[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %140 = llvm.insertvalue %127, %139[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %141 = llvm.insertvalue %127, %140[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %142 = llvm.insertvalue %128, %141[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %143 = llvm.mlir.constant(2 : index) : i64
        %144 = llvm.mlir.constant(3 : index) : i64
        %145 = llvm.mlir.constant(1 : index) : i64
        %146 = llvm.mlir.constant(6 : index) : i64
        %147 = llvm.mlir.null : !llvm.ptr
        %148 = llvm.getelementptr %147[%146] : (!llvm.ptr, i64) -> !llvm.ptr, i64
        %149 = llvm.ptrtoint %148 : !llvm.ptr to i64
        %150 = llvm.call @malloc(%149) : (i64) -> !llvm.ptr
        %151 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
        %152 = llvm.insertvalue %150, %151[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %153 = llvm.insertvalue %150, %152[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %154 = llvm.mlir.constant(0 : index) : i64
        %155 = llvm.insertvalue %154, %153[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %156 = llvm.insertvalue %143, %155[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %157 = llvm.insertvalue %144, %156[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %158 = llvm.insertvalue %144, %157[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %159 = llvm.insertvalue %145, %158[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %160 = llvm.mlir.constant(2 : index) : i64
        %161 = llvm.mlir.constant(3 : index) : i64
        %162 = llvm.mlir.constant(1 : index) : i64
        %163 = llvm.mlir.constant(6 : index) : i64
        %164 = llvm.mlir.null : !llvm.ptr
        %165 = llvm.getelementptr %164[%163] : (!llvm.ptr, i64) -> !llvm.ptr, i64
        %166 = llvm.ptrtoint %165 : !llvm.ptr to i64
        %167 = llvm.call @malloc(%166) : (i64) -> !llvm.ptr
        %168 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
        %169 = llvm.insertvalue %167, %168[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %170 = llvm.insertvalue %167, %169[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %171 = llvm.mlir.constant(0 : index) : i64
        %172 = llvm.insertvalue %171, %170[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %173 = llvm.insertvalue %160, %172[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %174 = llvm.insertvalue %161, %173[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %175 = llvm.insertvalue %161, %174[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %176 = llvm.insertvalue %162, %175[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %177 = llvm.mlir.constant(2 : index) : i64
        %178 = llvm.mlir.constant(3 : index) : i64
        %179 = llvm.mlir.constant(1 : index) : i64
        %180 = llvm.mlir.constant(6 : index) : i64
        %181 = llvm.mlir.null : !llvm.ptr
        %182 = llvm.getelementptr %181[%180] : (!llvm.ptr, i64) -> !llvm.ptr, f64
        %183 = llvm.ptrtoint %182 : !llvm.ptr to i64
        %184 = llvm.call @malloc(%183) : (i64) -> !llvm.ptr
        %185 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
        %186 = llvm.insertvalue %184, %185[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %187 = llvm.insertvalue %184, %186[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %188 = llvm.mlir.constant(0 : index) : i64
        %189 = llvm.insertvalue %188, %187[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %190 = llvm.insertvalue %177, %189[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %191 = llvm.insertvalue %178, %190[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %192 = llvm.insertvalue %178, %191[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %193 = llvm.insertvalue %179, %192[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %194 = llvm.mlir.constant(2 : index) : i64
        %195 = llvm.mlir.constant(3 : index) : i64
        %196 = llvm.mlir.constant(1 : index) : i64
        %197 = llvm.mlir.constant(6 : index) : i64
        %198 = llvm.mlir.null : !llvm.ptr
        %199 = llvm.getelementptr %198[%197] : (!llvm.ptr, i64) -> !llvm.ptr, f64
        %200 = llvm.ptrtoint %199 : !llvm.ptr to i64
        %201 = llvm.call @malloc(%200) : (i64) -> !llvm.ptr
        %202 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
        %203 = llvm.insertvalue %201, %202[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %204 = llvm.insertvalue %201, %203[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %205 = llvm.mlir.constant(0 : index) : i64
        %206 = llvm.insertvalue %205, %204[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %207 = llvm.insertvalue %194, %206[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %208 = llvm.insertvalue %195, %207[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %209 = llvm.insertvalue %195, %208[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %210 = llvm.insertvalue %196, %209[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %211 = llvm.mlir.constant(0 : index) : i64
        %212 = llvm.mlir.constant(0 : index) : i64
        %213 = llvm.extractvalue %210[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %214 = llvm.mlir.constant(3 : index) : i64
        %215 = llvm.mul %211, %214  : i64
        %216 = llvm.add %215, %212  : i64
        %217 = llvm.getelementptr %213[%216] : (!llvm.ptr, i64) -> !llvm.ptr, f64
        llvm.store %23, %217 : f64, !llvm.ptr
        %218 = llvm.mlir.constant(0 : index) : i64
        %219 = llvm.mlir.constant(1 : index) : i64
        %220 = llvm.extractvalue %210[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %221 = llvm.mlir.constant(3 : index) : i64
        %222 = llvm.mul %218, %221  : i64
        %223 = llvm.add %222, %219  : i64
        %224 = llvm.getelementptr %220[%223] : (!llvm.ptr, i64) -> !llvm.ptr, f64
        llvm.store %22, %224 : f64, !llvm.ptr
        %225 = llvm.mlir.constant(0 : index) : i64
        %226 = llvm.mlir.constant(2 : index) : i64
        %227 = llvm.extractvalue %210[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %228 = llvm.mlir.constant(3 : index) : i64
        %229 = llvm.mul %225, %228  : i64
        %230 = llvm.add %229, %226  : i64
        %231 = llvm.getelementptr %227[%230] : (!llvm.ptr, i64) -> !llvm.ptr, f64
        llvm.store %21, %231 : f64, !llvm.ptr
        %232 = llvm.mlir.constant(1 : index) : i64
        %233 = llvm.mlir.constant(0 : index) : i64
        %234 = llvm.extractvalue %210[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %235 = llvm.mlir.constant(3 : index) : i64
        %236 = llvm.mul %232, %235  : i64
        %237 = llvm.add %236, %233  : i64
        %238 = llvm.getelementptr %234[%237] : (!llvm.ptr, i64) -> !llvm.ptr, f64
        llvm.store %20, %238 : f64, !llvm.ptr
        %239 = llvm.mlir.constant(1 : index) : i64
        %240 = llvm.mlir.constant(1 : index) : i64
        %241 = llvm.extractvalue %210[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %242 = llvm.mlir.constant(3 : index) : i64
        %243 = llvm.mul %239, %242  : i64
        %244 = llvm.add %243, %240  : i64
        %245 = llvm.getelementptr %241[%244] : (!llvm.ptr, i64) -> !llvm.ptr, f64
        llvm.store %19, %245 : f64, !llvm.ptr
        %246 = llvm.mlir.constant(1 : index) : i64
        %247 = llvm.mlir.constant(2 : index) : i64
        %248 = llvm.extractvalue %210[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %249 = llvm.mlir.constant(3 : index) : i64
        %250 = llvm.mul %246, %249  : i64
        %251 = llvm.add %250, %247  : i64
        %252 = llvm.getelementptr %248[%251] : (!llvm.ptr, i64) -> !llvm.ptr, f64
        llvm.store %18, %252 : f64, !llvm.ptr
        %253 = llvm.mlir.constant(0 : index) : i64
        %254 = llvm.mlir.constant(0 : index) : i64
        %255 = llvm.extractvalue %193[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %256 = llvm.mlir.constant(3 : index) : i64
        %257 = llvm.mul %253, %256  : i64
        %258 = llvm.add %257, %254  : i64
        %259 = llvm.getelementptr %255[%258] : (!llvm.ptr, i64) -> !llvm.ptr, f64
        llvm.store %17, %259 : f64, !llvm.ptr
        %260 = llvm.mlir.constant(0 : index) : i64
        %261 = llvm.mlir.constant(1 : index) : i64
        %262 = llvm.extractvalue %193[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %263 = llvm.mlir.constant(3 : index) : i64
        %264 = llvm.mul %260, %263  : i64
        %265 = llvm.add %264, %261  : i64
        %266 = llvm.getelementptr %262[%265] : (!llvm.ptr, i64) -> !llvm.ptr, f64
        llvm.store %16, %266 : f64, !llvm.ptr
        %267 = llvm.mlir.constant(0 : index) : i64
        %268 = llvm.mlir.constant(2 : index) : i64
        %269 = llvm.extractvalue %193[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %270 = llvm.mlir.constant(3 : index) : i64
        %271 = llvm.mul %267, %270  : i64
        %272 = llvm.add %271, %268  : i64
        %273 = llvm.getelementptr %269[%272] : (!llvm.ptr, i64) -> !llvm.ptr, f64
        llvm.store %15, %273 : f64, !llvm.ptr
        %274 = llvm.mlir.constant(1 : index) : i64
        %275 = llvm.mlir.constant(0 : index) : i64
        %276 = llvm.extractvalue %193[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %277 = llvm.mlir.constant(3 : index) : i64
        %278 = llvm.mul %274, %277  : i64
        %279 = llvm.add %278, %275  : i64
        %280 = llvm.getelementptr %276[%279] : (!llvm.ptr, i64) -> !llvm.ptr, f64
        llvm.store %14, %280 : f64, !llvm.ptr
        %281 = llvm.mlir.constant(1 : index) : i64
        %282 = llvm.mlir.constant(1 : index) : i64
        %283 = llvm.extractvalue %193[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %284 = llvm.mlir.constant(3 : index) : i64
        %285 = llvm.mul %281, %284  : i64
        %286 = llvm.add %285, %282  : i64
        %287 = llvm.getelementptr %283[%286] : (!llvm.ptr, i64) -> !llvm.ptr, f64
        llvm.store %13, %287 : f64, !llvm.ptr
        %288 = llvm.mlir.constant(1 : index) : i64
        %289 = llvm.mlir.constant(2 : index) : i64
        %290 = llvm.extractvalue %193[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %291 = llvm.mlir.constant(3 : index) : i64
        %292 = llvm.mul %288, %291  : i64
        %293 = llvm.add %292, %289  : i64
        %294 = llvm.getelementptr %290[%293] : (!llvm.ptr, i64) -> !llvm.ptr, f64
        llvm.store %12, %294 : f64, !llvm.ptr
        %295 = llvm.mlir.constant(0 : index) : i64
        %296 = llvm.mlir.constant(0 : index) : i64
        %297 = llvm.extractvalue %176[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %298 = llvm.mlir.constant(3 : index) : i64
        %299 = llvm.mul %295, %298  : i64
        %300 = llvm.add %299, %296  : i64
        %301 = llvm.getelementptr %297[%300] : (!llvm.ptr, i64) -> !llvm.ptr, i64
        llvm.store %11, %301 : i64, !llvm.ptr
        %302 = llvm.mlir.constant(0 : index) : i64
        %303 = llvm.mlir.constant(1 : index) : i64
        %304 = llvm.extractvalue %176[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %305 = llvm.mlir.constant(3 : index) : i64
        %306 = llvm.mul %302, %305  : i64
        %307 = llvm.add %306, %303  : i64
        %308 = llvm.getelementptr %304[%307] : (!llvm.ptr, i64) -> !llvm.ptr, i64
        llvm.store %10, %308 : i64, !llvm.ptr
        %309 = llvm.mlir.constant(0 : index) : i64
        %310 = llvm.mlir.constant(2 : index) : i64
        %311 = llvm.extractvalue %176[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %312 = llvm.mlir.constant(3 : index) : i64
        %313 = llvm.mul %309, %312  : i64
        %314 = llvm.add %313, %310  : i64
        %315 = llvm.getelementptr %311[%314] : (!llvm.ptr, i64) -> !llvm.ptr, i64
        llvm.store %9, %315 : i64, !llvm.ptr
        %316 = llvm.mlir.constant(1 : index) : i64
        %317 = llvm.mlir.constant(0 : index) : i64
        %318 = llvm.extractvalue %176[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %319 = llvm.mlir.constant(3 : index) : i64
        %320 = llvm.mul %316, %319  : i64
        %321 = llvm.add %320, %317  : i64
        %322 = llvm.getelementptr %318[%321] : (!llvm.ptr, i64) -> !llvm.ptr, i64
        llvm.store %8, %322 : i64, !llvm.ptr
        %323 = llvm.mlir.constant(1 : index) : i64
        %324 = llvm.mlir.constant(1 : index) : i64
        %325 = llvm.extractvalue %176[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %326 = llvm.mlir.constant(3 : index) : i64
        %327 = llvm.mul %323, %326  : i64
        %328 = llvm.add %327, %324  : i64
        %329 = llvm.getelementptr %325[%328] : (!llvm.ptr, i64) -> !llvm.ptr, i64
        llvm.store %7, %329 : i64, !llvm.ptr
        %330 = llvm.mlir.constant(1 : index) : i64
        %331 = llvm.mlir.constant(2 : index) : i64
        %332 = llvm.extractvalue %176[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %333 = llvm.mlir.constant(3 : index) : i64
        %334 = llvm.mul %330, %333  : i64
        %335 = llvm.add %334, %331  : i64
        %336 = llvm.getelementptr %332[%335] : (!llvm.ptr, i64) -> !llvm.ptr, i64
        llvm.store %6, %336 : i64, !llvm.ptr
        %337 = llvm.mlir.constant(0 : index) : i64
        %338 = llvm.mlir.constant(0 : index) : i64
        %339 = llvm.extractvalue %159[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %340 = llvm.mlir.constant(3 : index) : i64
        %341 = llvm.mul %337, %340  : i64
        %342 = llvm.add %341, %338  : i64
        %343 = llvm.getelementptr %339[%342] : (!llvm.ptr, i64) -> !llvm.ptr, i64
        llvm.store %5, %343 : i64, !llvm.ptr
        %344 = llvm.mlir.constant(0 : index) : i64
        %345 = llvm.mlir.constant(1 : index) : i64
        %346 = llvm.extractvalue %159[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %347 = llvm.mlir.constant(3 : index) : i64
        %348 = llvm.mul %344, %347  : i64
        %349 = llvm.add %348, %345  : i64
        %350 = llvm.getelementptr %346[%349] : (!llvm.ptr, i64) -> !llvm.ptr, i64
        llvm.store %4, %350 : i64, !llvm.ptr
        %351 = llvm.mlir.constant(0 : index) : i64
        %352 = llvm.mlir.constant(2 : index) : i64
        %353 = llvm.extractvalue %159[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %354 = llvm.mlir.constant(3 : index) : i64
        %355 = llvm.mul %351, %354  : i64
        %356 = llvm.add %355, %352  : i64
        %357 = llvm.getelementptr %353[%356] : (!llvm.ptr, i64) -> !llvm.ptr, i64
        llvm.store %3, %357 : i64, !llvm.ptr
        %358 = llvm.mlir.constant(1 : index) : i64
        %359 = llvm.mlir.constant(0 : index) : i64
        %360 = llvm.extractvalue %159[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %361 = llvm.mlir.constant(3 : index) : i64
        %362 = llvm.mul %358, %361  : i64
        %363 = llvm.add %362, %359  : i64
        %364 = llvm.getelementptr %360[%363] : (!llvm.ptr, i64) -> !llvm.ptr, i64
        llvm.store %2, %364 : i64, !llvm.ptr
        %365 = llvm.mlir.constant(1 : index) : i64
        %366 = llvm.mlir.constant(1 : index) : i64
        %367 = llvm.extractvalue %159[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %368 = llvm.mlir.constant(3 : index) : i64
        %369 = llvm.mul %365, %368  : i64
        %370 = llvm.add %369, %366  : i64
        %371 = llvm.getelementptr %367[%370] : (!llvm.ptr, i64) -> !llvm.ptr, i64
        llvm.store %1, %371 : i64, !llvm.ptr
        %372 = llvm.mlir.constant(1 : index) : i64
        %373 = llvm.mlir.constant(2 : index) : i64
        %374 = llvm.extractvalue %159[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %375 = llvm.mlir.constant(3 : index) : i64
        %376 = llvm.mul %372, %375  : i64
        %377 = llvm.add %376, %373  : i64
        %378 = llvm.getelementptr %374[%377] : (!llvm.ptr, i64) -> !llvm.ptr, i64
        llvm.store %0, %378 : i64, !llvm.ptr
        %379 = llvm.mlir.constant(0 : index) : i64
        %380 = llvm.mlir.constant(3 : index) : i64
        %381 = llvm.mlir.constant(1 : index) : i64
        llvm.br ^bb1(%379 : i64)
      ^bb1(%382: i64):  // 2 preds: ^bb0, ^bb5
        %383 = llvm.icmp "slt" %382, %380 : i64
        llvm.cond_br %383, ^bb2, ^bb6
      ^bb2:  // pred: ^bb1
        %384 = llvm.mlir.constant(0 : index) : i64
        %385 = llvm.mlir.constant(2 : index) : i64
        %386 = llvm.mlir.constant(1 : index) : i64
        llvm.br ^bb3(%384 : i64)
      ^bb3(%387: i64):  // 2 preds: ^bb2, ^bb4
        %388 = llvm.icmp "slt" %387, %385 : i64
        llvm.cond_br %388, ^bb4, ^bb5
      ^bb4:  // pred: ^bb3
        %389 = llvm.extractvalue %210[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %390 = llvm.mlir.constant(3 : index) : i64
        %391 = llvm.mul %387, %390  : i64
        %392 = llvm.add %391, %382  : i64
        %393 = llvm.getelementptr %389[%392] : (!llvm.ptr, i64) -> !llvm.ptr, f64
        %394 = llvm.load %393 : !llvm.ptr -> f64
        %395 = llvm.extractvalue %193[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %396 = llvm.mlir.constant(3 : index) : i64
        %397 = llvm.mul %387, %396  : i64
        %398 = llvm.add %397, %382  : i64
        %399 = llvm.getelementptr %395[%398] : (!llvm.ptr, i64) -> !llvm.ptr, f64
        %400 = llvm.load %399 : !llvm.ptr -> f64
        %401 = llvm.fmul %394, %400  : f64
        %402 = llvm.extractvalue %142[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %403 = llvm.mlir.constant(2 : index) : i64
        %404 = llvm.mul %382, %403  : i64
        %405 = llvm.add %404, %387  : i64
        %406 = llvm.getelementptr %402[%405] : (!llvm.ptr, i64) -> !llvm.ptr, f64
        llvm.store %401, %406 : f64, !llvm.ptr
        %407 = llvm.fmul %400, %394  : f64
        %408 = llvm.extractvalue %125[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %409 = llvm.mlir.constant(2 : index) : i64
        %410 = llvm.mul %382, %409  : i64
        %411 = llvm.add %410, %387  : i64
        %412 = llvm.getelementptr %408[%411] : (!llvm.ptr, i64) -> !llvm.ptr, f64
        llvm.store %407, %412 : f64, !llvm.ptr
        %413 = llvm.add %387, %386  : i64
        llvm.br ^bb3(%413 : i64)
      ^bb5:  // pred: ^bb3
        %414 = llvm.add %382, %381  : i64
        llvm.br ^bb1(%414 : i64)
      ^bb6:  // pred: ^bb1
        %415 = llvm.mlir.addressof @int_frmt_spec : !llvm.ptr<array<4 x i8>>
        %416 = llvm.mlir.constant(0 : index) : i64
        %417 = llvm.getelementptr %415[%416, %416] : (!llvm.ptr<array<4 x i8>>, i64, i64) -> !llvm.ptr<i8>
        %418 = llvm.mlir.addressof @float_frmt_spec : !llvm.ptr<array<4 x i8>>
        %419 = llvm.mlir.constant(0 : index) : i64
        %420 = llvm.getelementptr %418[%419, %419] : (!llvm.ptr<array<4 x i8>>, i64, i64) -> !llvm.ptr<i8>
        %421 = llvm.mlir.addressof @nl : !llvm.ptr<array<2 x i8>>
        %422 = llvm.mlir.constant(0 : index) : i64
        %423 = llvm.getelementptr %421[%422, %422] : (!llvm.ptr<array<2 x i8>>, i64, i64) -> !llvm.ptr<i8>
        %424 = llvm.mlir.constant(0 : index) : i64
        %425 = llvm.mlir.constant(3 : index) : i64
        %426 = llvm.mlir.constant(1 : index) : i64
        llvm.br ^bb7(%424 : i64)
      ^bb7(%427: i64):  // 2 preds: ^bb6, ^bb11
        %428 = llvm.icmp "slt" %427, %425 : i64
        llvm.cond_br %428, ^bb8, ^bb12
      ^bb8:  // pred: ^bb7
        %429 = llvm.mlir.constant(0 : index) : i64
        %430 = llvm.mlir.constant(2 : index) : i64
        %431 = llvm.mlir.constant(1 : index) : i64
        llvm.br ^bb9(%429 : i64)
      ^bb9(%432: i64):  // 2 preds: ^bb8, ^bb10
        %433 = llvm.icmp "slt" %432, %430 : i64
        llvm.cond_br %433, ^bb10, ^bb11
      ^bb10:  // pred: ^bb9
        %434 = llvm.extractvalue %142[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %435 = llvm.mlir.constant(2 : index) : i64
        %436 = llvm.mul %427, %435  : i64
        %437 = llvm.add %436, %432  : i64
        %438 = llvm.getelementptr %434[%437] : (!llvm.ptr, i64) -> !llvm.ptr, f64
        %439 = llvm.load %438 : !llvm.ptr -> f64
        %440 = llvm.call @printf(%420, %439) : (!llvm.ptr<i8>, f64) -> i32
        %441 = llvm.add %432, %431  : i64
        llvm.br ^bb9(%441 : i64)
      ^bb11:  // pred: ^bb9
        %442 = llvm.call @printf(%423) : (!llvm.ptr<i8>) -> i32
        %443 = llvm.add %427, %426  : i64
        llvm.br ^bb7(%443 : i64)
      ^bb12:  // pred: ^bb7
        %444 = llvm.mlir.addressof @int_frmt_spec : !llvm.ptr<array<4 x i8>>
        %445 = llvm.mlir.constant(0 : index) : i64
        %446 = llvm.getelementptr %444[%445, %445] : (!llvm.ptr<array<4 x i8>>, i64, i64) -> !llvm.ptr<i8>
        %447 = llvm.mlir.addressof @float_frmt_spec : !llvm.ptr<array<4 x i8>>
        %448 = llvm.mlir.constant(0 : index) : i64
        %449 = llvm.getelementptr %447[%448, %448] : (!llvm.ptr<array<4 x i8>>, i64, i64) -> !llvm.ptr<i8>
        %450 = llvm.mlir.addressof @nl : !llvm.ptr<array<2 x i8>>
        %451 = llvm.mlir.constant(0 : index) : i64
        %452 = llvm.getelementptr %450[%451, %451] : (!llvm.ptr<array<2 x i8>>, i64, i64) -> !llvm.ptr<i8>
        %453 = llvm.mlir.constant(0 : index) : i64
        %454 = llvm.mlir.constant(3 : index) : i64
        %455 = llvm.mlir.constant(1 : index) : i64
        llvm.br ^bb13(%453 : i64)
      ^bb13(%456: i64):  // 2 preds: ^bb12, ^bb17
        %457 = llvm.icmp "slt" %456, %454 : i64
        llvm.cond_br %457, ^bb14, ^bb18
      ^bb14:  // pred: ^bb13
        %458 = llvm.mlir.constant(0 : index) : i64
        %459 = llvm.mlir.constant(2 : index) : i64
        %460 = llvm.mlir.constant(1 : index) : i64
        llvm.br ^bb15(%458 : i64)
      ^bb15(%461: i64):  // 2 preds: ^bb14, ^bb16
        %462 = llvm.icmp "slt" %461, %459 : i64
        llvm.cond_br %462, ^bb16, ^bb17
      ^bb16:  // pred: ^bb15
        %463 = llvm.extractvalue %125[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %464 = llvm.mlir.constant(2 : index) : i64
        %465 = llvm.mul %456, %464  : i64
        %466 = llvm.add %465, %461  : i64
        %467 = llvm.getelementptr %463[%466] : (!llvm.ptr, i64) -> !llvm.ptr, f64
        %468 = llvm.load %467 : !llvm.ptr -> f64
        %469 = llvm.call @printf(%449, %468) : (!llvm.ptr<i8>, f64) -> i32
        %470 = llvm.add %461, %460  : i64
        llvm.br ^bb15(%470 : i64)
      ^bb17:  // pred: ^bb15
        %471 = llvm.call @printf(%452) : (!llvm.ptr<i8>) -> i32
        %472 = llvm.add %456, %455  : i64
        llvm.br ^bb13(%472 : i64)
      ^bb18:  // pred: ^bb13
        %473 = llvm.mlir.constant(0 : index) : i64
        %474 = llvm.mlir.constant(2 : index) : i64
        %475 = llvm.mlir.constant(1 : index) : i64
        llvm.br ^bb19(%473 : i64)
      ^bb19(%476: i64):  // 2 preds: ^bb18, ^bb23
        %477 = llvm.icmp "slt" %476, %474 : i64
        llvm.cond_br %477, ^bb20, ^bb24
      ^bb20:  // pred: ^bb19
        %478 = llvm.mlir.constant(0 : index) : i64
        %479 = llvm.mlir.constant(3 : index) : i64
        %480 = llvm.mlir.constant(1 : index) : i64
        llvm.br ^bb21(%478 : i64)
      ^bb21(%481: i64):  // 2 preds: ^bb20, ^bb22
        %482 = llvm.icmp "slt" %481, %479 : i64
        llvm.cond_br %482, ^bb22, ^bb23
      ^bb22:  // pred: ^bb21
        %483 = llvm.extractvalue %176[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %484 = llvm.mlir.constant(3 : index) : i64
        %485 = llvm.mul %476, %484  : i64
        %486 = llvm.add %485, %481  : i64
        %487 = llvm.getelementptr %483[%486] : (!llvm.ptr, i64) -> !llvm.ptr, i64
        %488 = llvm.load %487 : !llvm.ptr -> i64
        %489 = llvm.extractvalue %159[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %490 = llvm.mlir.constant(3 : index) : i64
        %491 = llvm.mul %476, %490  : i64
        %492 = llvm.add %491, %481  : i64
        %493 = llvm.getelementptr %489[%492] : (!llvm.ptr, i64) -> !llvm.ptr, i64
        %494 = llvm.load %493 : !llvm.ptr -> i64
        %495 = llvm.or %488, %494  : i64
        %496 = llvm.extractvalue %108[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %497 = llvm.mlir.constant(3 : index) : i64
        %498 = llvm.mul %476, %497  : i64
        %499 = llvm.add %498, %481  : i64
        %500 = llvm.getelementptr %496[%499] : (!llvm.ptr, i64) -> !llvm.ptr, i64
        llvm.store %495, %500 : i64, !llvm.ptr
        %501 = llvm.add %481, %480  : i64
        llvm.br ^bb21(%501 : i64)
      ^bb23:  // pred: ^bb21
        %502 = llvm.add %476, %475  : i64
        llvm.br ^bb19(%502 : i64)
      ^bb24:  // pred: ^bb19
        %503 = llvm.mlir.constant(0 : index) : i64
        %504 = llvm.mlir.constant(2 : index) : i64
        %505 = llvm.mlir.constant(1 : index) : i64
        llvm.br ^bb25(%503 : i64)
      ^bb25(%506: i64):  // 2 preds: ^bb24, ^bb29
        %507 = llvm.icmp "slt" %506, %504 : i64
        llvm.cond_br %507, ^bb26, ^bb30
      ^bb26:  // pred: ^bb25
        %508 = llvm.mlir.constant(0 : index) : i64
        %509 = llvm.mlir.constant(3 : index) : i64
        %510 = llvm.mlir.constant(1 : index) : i64
        llvm.br ^bb27(%508 : i64)
      ^bb27(%511: i64):  // 2 preds: ^bb26, ^bb28
        %512 = llvm.icmp "slt" %511, %509 : i64
        llvm.cond_br %512, ^bb28, ^bb29
      ^bb28:  // pred: ^bb27
        %513 = llvm.extractvalue %176[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %514 = llvm.mlir.constant(3 : index) : i64
        %515 = llvm.mul %506, %514  : i64
        %516 = llvm.add %515, %511  : i64
        %517 = llvm.getelementptr %513[%516] : (!llvm.ptr, i64) -> !llvm.ptr, i64
        %518 = llvm.load %517 : !llvm.ptr -> i64
        %519 = llvm.extractvalue %159[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %520 = llvm.mlir.constant(3 : index) : i64
        %521 = llvm.mul %506, %520  : i64
        %522 = llvm.add %521, %511  : i64
        %523 = llvm.getelementptr %519[%522] : (!llvm.ptr, i64) -> !llvm.ptr, i64
        %524 = llvm.load %523 : !llvm.ptr -> i64
        %525 = llvm.and %518, %524  : i64
        %526 = llvm.extractvalue %91[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %527 = llvm.mlir.constant(3 : index) : i64
        %528 = llvm.mul %506, %527  : i64
        %529 = llvm.add %528, %511  : i64
        %530 = llvm.getelementptr %526[%529] : (!llvm.ptr, i64) -> !llvm.ptr, i64
        llvm.store %525, %530 : i64, !llvm.ptr
        %531 = llvm.add %511, %510  : i64
        llvm.br ^bb27(%531 : i64)
      ^bb29:  // pred: ^bb27
        %532 = llvm.add %506, %505  : i64
        llvm.br ^bb25(%532 : i64)
      ^bb30:  // pred: ^bb25
        %533 = llvm.mlir.addressof @int_frmt_spec : !llvm.ptr<array<4 x i8>>
        %534 = llvm.mlir.constant(0 : index) : i64
        %535 = llvm.getelementptr %533[%534, %534] : (!llvm.ptr<array<4 x i8>>, i64, i64) -> !llvm.ptr<i8>
        %536 = llvm.mlir.addressof @float_frmt_spec : !llvm.ptr<array<4 x i8>>
        %537 = llvm.mlir.constant(0 : index) : i64
        %538 = llvm.getelementptr %536[%537, %537] : (!llvm.ptr<array<4 x i8>>, i64, i64) -> !llvm.ptr<i8>
        %539 = llvm.mlir.addressof @nl : !llvm.ptr<array<2 x i8>>
        %540 = llvm.mlir.constant(0 : index) : i64
        %541 = llvm.getelementptr %539[%540, %540] : (!llvm.ptr<array<2 x i8>>, i64, i64) -> !llvm.ptr<i8>
        %542 = llvm.mlir.constant(0 : index) : i64
        %543 = llvm.mlir.constant(2 : index) : i64
        %544 = llvm.mlir.constant(1 : index) : i64
        llvm.br ^bb31(%542 : i64)
      ^bb31(%545: i64):  // 2 preds: ^bb30, ^bb35
        %546 = llvm.icmp "slt" %545, %543 : i64
        llvm.cond_br %546, ^bb32, ^bb36
      ^bb32:  // pred: ^bb31
        %547 = llvm.mlir.constant(0 : index) : i64
        %548 = llvm.mlir.constant(3 : index) : i64
        %549 = llvm.mlir.constant(1 : index) : i64
        llvm.br ^bb33(%547 : i64)
      ^bb33(%550: i64):  // 2 preds: ^bb32, ^bb34
        %551 = llvm.icmp "slt" %550, %548 : i64
        llvm.cond_br %551, ^bb34, ^bb35
      ^bb34:  // pred: ^bb33
        %552 = llvm.extractvalue %108[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %553 = llvm.mlir.constant(3 : index) : i64
        %554 = llvm.mul %545, %553  : i64
        %555 = llvm.add %554, %550  : i64
        %556 = llvm.getelementptr %552[%555] : (!llvm.ptr, i64) -> !llvm.ptr, i64
        %557 = llvm.load %556 : !llvm.ptr -> i64
        %558 = llvm.call @printf(%535, %557) : (!llvm.ptr<i8>, i64) -> i32
        %559 = llvm.add %550, %549  : i64
        llvm.br ^bb33(%559 : i64)
      ^bb35:  // pred: ^bb33
        %560 = llvm.call @printf(%541) : (!llvm.ptr<i8>) -> i32
        %561 = llvm.add %545, %544  : i64
        llvm.br ^bb31(%561 : i64)
      ^bb36:  // pred: ^bb31
        %562 = llvm.mlir.addressof @int_frmt_spec : !llvm.ptr<array<4 x i8>>
        %563 = llvm.mlir.constant(0 : index) : i64
        %564 = llvm.getelementptr %562[%563, %563] : (!llvm.ptr<array<4 x i8>>, i64, i64) -> !llvm.ptr<i8>
        %565 = llvm.mlir.addressof @float_frmt_spec : !llvm.ptr<array<4 x i8>>
        %566 = llvm.mlir.constant(0 : index) : i64
        %567 = llvm.getelementptr %565[%566, %566] : (!llvm.ptr<array<4 x i8>>, i64, i64) -> !llvm.ptr<i8>
        %568 = llvm.mlir.addressof @nl : !llvm.ptr<array<2 x i8>>
        %569 = llvm.mlir.constant(0 : index) : i64
        %570 = llvm.getelementptr %568[%569, %569] : (!llvm.ptr<array<2 x i8>>, i64, i64) -> !llvm.ptr<i8>
        %571 = llvm.mlir.constant(0 : index) : i64
        %572 = llvm.mlir.constant(2 : index) : i64
        %573 = llvm.mlir.constant(1 : index) : i64
        llvm.br ^bb37(%571 : i64)
      ^bb37(%574: i64):  // 2 preds: ^bb36, ^bb41
        %575 = llvm.icmp "slt" %574, %572 : i64
        llvm.cond_br %575, ^bb38, ^bb42
      ^bb38:  // pred: ^bb37
        %576 = llvm.mlir.constant(0 : index) : i64
        %577 = llvm.mlir.constant(3 : index) : i64
        %578 = llvm.mlir.constant(1 : index) : i64
        llvm.br ^bb39(%576 : i64)
      ^bb39(%579: i64):  // 2 preds: ^bb38, ^bb40
        %580 = llvm.icmp "slt" %579, %577 : i64
        llvm.cond_br %580, ^bb40, ^bb41
      ^bb40:  // pred: ^bb39
        %581 = llvm.extractvalue %91[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %582 = llvm.mlir.constant(3 : index) : i64
        %583 = llvm.mul %574, %582  : i64
        %584 = llvm.add %583, %579  : i64
        %585 = llvm.getelementptr %581[%584] : (!llvm.ptr, i64) -> !llvm.ptr, i64
        %586 = llvm.load %585 : !llvm.ptr -> i64
        %587 = llvm.call @printf(%564, %586) : (!llvm.ptr<i8>, i64) -> i32
        %588 = llvm.add %579, %578  : i64
        llvm.br ^bb39(%588 : i64)
      ^bb41:  // pred: ^bb39
        %589 = llvm.call @printf(%570) : (!llvm.ptr<i8>) -> i32
        %590 = llvm.add %574, %573  : i64
        llvm.br ^bb37(%590 : i64)
      ^bb42:  // pred: ^bb37
        %591 = llvm.mlir.constant(0 : index) : i64
        %592 = llvm.mlir.constant(2 : index) : i64
        %593 = llvm.mlir.constant(1 : index) : i64
        llvm.br ^bb43(%591 : i64)
      ^bb43(%594: i64):  // 2 preds: ^bb42, ^bb47
        %595 = llvm.icmp "slt" %594, %592 : i64
        llvm.cond_br %595, ^bb44, ^bb48
      ^bb44:  // pred: ^bb43
        %596 = llvm.mlir.constant(0 : index) : i64
        %597 = llvm.mlir.constant(3 : index) : i64
        %598 = llvm.mlir.constant(1 : index) : i64
        llvm.br ^bb45(%596 : i64)
      ^bb45(%599: i64):  // 2 preds: ^bb44, ^bb46
        %600 = llvm.icmp "slt" %599, %597 : i64
        llvm.cond_br %600, ^bb46, ^bb47
      ^bb46:  // pred: ^bb45
        %601 = llvm.extractvalue %210[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %602 = llvm.mlir.constant(3 : index) : i64
        %603 = llvm.mul %594, %602  : i64
        %604 = llvm.add %603, %599  : i64
        %605 = llvm.getelementptr %601[%604] : (!llvm.ptr, i64) -> !llvm.ptr, f64
        %606 = llvm.load %605 : !llvm.ptr -> f64
        %607 = llvm.extractvalue %193[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %608 = llvm.mlir.constant(3 : index) : i64
        %609 = llvm.mul %594, %608  : i64
        %610 = llvm.add %609, %599  : i64
        %611 = llvm.getelementptr %607[%610] : (!llvm.ptr, i64) -> !llvm.ptr, f64
        %612 = llvm.load %611 : !llvm.ptr -> f64
        %613 = llvm.fadd %606, %612  : f64
        %614 = llvm.extractvalue %74[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %615 = llvm.mlir.constant(3 : index) : i64
        %616 = llvm.mul %594, %615  : i64
        %617 = llvm.add %616, %599  : i64
        %618 = llvm.getelementptr %614[%617] : (!llvm.ptr, i64) -> !llvm.ptr, f64
        llvm.store %613, %618 : f64, !llvm.ptr
        %619 = llvm.add %599, %598  : i64
        llvm.br ^bb45(%619 : i64)
      ^bb47:  // pred: ^bb45
        %620 = llvm.add %594, %593  : i64
        llvm.br ^bb43(%620 : i64)
      ^bb48:  // pred: ^bb43
        %621 = llvm.mlir.constant(0 : index) : i64
        %622 = llvm.mlir.constant(2 : index) : i64
        %623 = llvm.mlir.constant(1 : index) : i64
        llvm.br ^bb49(%621 : i64)
      ^bb49(%624: i64):  // 2 preds: ^bb48, ^bb53
        %625 = llvm.icmp "slt" %624, %622 : i64
        llvm.cond_br %625, ^bb50, ^bb54
      ^bb50:  // pred: ^bb49
        %626 = llvm.mlir.constant(0 : index) : i64
        %627 = llvm.mlir.constant(3 : index) : i64
        %628 = llvm.mlir.constant(1 : index) : i64
        llvm.br ^bb51(%626 : i64)
      ^bb51(%629: i64):  // 2 preds: ^bb50, ^bb52
        %630 = llvm.icmp "slt" %629, %627 : i64
        llvm.cond_br %630, ^bb52, ^bb53
      ^bb52:  // pred: ^bb51
        %631 = llvm.extractvalue %210[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %632 = llvm.mlir.constant(3 : index) : i64
        %633 = llvm.mul %624, %632  : i64
        %634 = llvm.add %633, %629  : i64
        %635 = llvm.getelementptr %631[%634] : (!llvm.ptr, i64) -> !llvm.ptr, f64
        %636 = llvm.load %635 : !llvm.ptr -> f64
        %637 = llvm.extractvalue %193[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %638 = llvm.mlir.constant(3 : index) : i64
        %639 = llvm.mul %624, %638  : i64
        %640 = llvm.add %639, %629  : i64
        %641 = llvm.getelementptr %637[%640] : (!llvm.ptr, i64) -> !llvm.ptr, f64
        %642 = llvm.load %641 : !llvm.ptr -> f64
        %643 = llvm.fsub %636, %642  : f64
        %644 = llvm.extractvalue %57[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %645 = llvm.mlir.constant(3 : index) : i64
        %646 = llvm.mul %624, %645  : i64
        %647 = llvm.add %646, %629  : i64
        %648 = llvm.getelementptr %644[%647] : (!llvm.ptr, i64) -> !llvm.ptr, f64
        llvm.store %643, %648 : f64, !llvm.ptr
        %649 = llvm.add %629, %628  : i64
        llvm.br ^bb51(%649 : i64)
      ^bb53:  // pred: ^bb51
        %650 = llvm.add %624, %623  : i64
        llvm.br ^bb49(%650 : i64)
      ^bb54:  // pred: ^bb49
        %651 = llvm.mlir.constant(0 : index) : i64
        %652 = llvm.mlir.constant(2 : index) : i64
        %653 = llvm.mlir.constant(1 : index) : i64
        llvm.br ^bb55(%651 : i64)
      ^bb55(%654: i64):  // 2 preds: ^bb54, ^bb59
        %655 = llvm.icmp "slt" %654, %652 : i64
        llvm.cond_br %655, ^bb56, ^bb60
      ^bb56:  // pred: ^bb55
        %656 = llvm.mlir.constant(0 : index) : i64
        %657 = llvm.mlir.constant(3 : index) : i64
        %658 = llvm.mlir.constant(1 : index) : i64
        llvm.br ^bb57(%656 : i64)
      ^bb57(%659: i64):  // 2 preds: ^bb56, ^bb58
        %660 = llvm.icmp "slt" %659, %657 : i64
        llvm.cond_br %660, ^bb58, ^bb59
      ^bb58:  // pred: ^bb57
        %661 = llvm.extractvalue %210[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %662 = llvm.mlir.constant(3 : index) : i64
        %663 = llvm.mul %654, %662  : i64
        %664 = llvm.add %663, %659  : i64
        %665 = llvm.getelementptr %661[%664] : (!llvm.ptr, i64) -> !llvm.ptr, f64
        %666 = llvm.load %665 : !llvm.ptr -> f64
        %667 = llvm.extractvalue %193[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %668 = llvm.mlir.constant(3 : index) : i64
        %669 = llvm.mul %654, %668  : i64
        %670 = llvm.add %669, %659  : i64
        %671 = llvm.getelementptr %667[%670] : (!llvm.ptr, i64) -> !llvm.ptr, f64
        %672 = llvm.load %671 : !llvm.ptr -> f64
        %673 = llvm.fmul %666, %672  : f64
        %674 = llvm.extractvalue %40[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %675 = llvm.mlir.constant(3 : index) : i64
        %676 = llvm.mul %654, %675  : i64
        %677 = llvm.add %676, %659  : i64
        %678 = llvm.getelementptr %674[%677] : (!llvm.ptr, i64) -> !llvm.ptr, f64
        llvm.store %673, %678 : f64, !llvm.ptr
        %679 = llvm.add %659, %658  : i64
        llvm.br ^bb57(%679 : i64)
      ^bb59:  // pred: ^bb57
        %680 = llvm.add %654, %653  : i64
        llvm.br ^bb55(%680 : i64)
      ^bb60:  // pred: ^bb55
        %681 = llvm.mlir.addressof @int_frmt_spec : !llvm.ptr<array<4 x i8>>
        %682 = llvm.mlir.constant(0 : index) : i64
        %683 = llvm.getelementptr %681[%682, %682] : (!llvm.ptr<array<4 x i8>>, i64, i64) -> !llvm.ptr<i8>
        %684 = llvm.mlir.addressof @float_frmt_spec : !llvm.ptr<array<4 x i8>>
        %685 = llvm.mlir.constant(0 : index) : i64
        %686 = llvm.getelementptr %684[%685, %685] : (!llvm.ptr<array<4 x i8>>, i64, i64) -> !llvm.ptr<i8>
        %687 = llvm.mlir.addressof @nl : !llvm.ptr<array<2 x i8>>
        %688 = llvm.mlir.constant(0 : index) : i64
        %689 = llvm.getelementptr %687[%688, %688] : (!llvm.ptr<array<2 x i8>>, i64, i64) -> !llvm.ptr<i8>
        %690 = llvm.mlir.constant(0 : index) : i64
        %691 = llvm.mlir.constant(2 : index) : i64
        %692 = llvm.mlir.constant(1 : index) : i64
        llvm.br ^bb61(%690 : i64)
      ^bb61(%693: i64):  // 2 preds: ^bb60, ^bb65
        %694 = llvm.icmp "slt" %693, %691 : i64
        llvm.cond_br %694, ^bb62, ^bb66
      ^bb62:  // pred: ^bb61
        %695 = llvm.mlir.constant(0 : index) : i64
        %696 = llvm.mlir.constant(3 : index) : i64
        %697 = llvm.mlir.constant(1 : index) : i64
        llvm.br ^bb63(%695 : i64)
      ^bb63(%698: i64):  // 2 preds: ^bb62, ^bb64
        %699 = llvm.icmp "slt" %698, %696 : i64
        llvm.cond_br %699, ^bb64, ^bb65
      ^bb64:  // pred: ^bb63
        %700 = llvm.extractvalue %74[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %701 = llvm.mlir.constant(3 : index) : i64
        %702 = llvm.mul %693, %701  : i64
        %703 = llvm.add %702, %698  : i64
        %704 = llvm.getelementptr %700[%703] : (!llvm.ptr, i64) -> !llvm.ptr, f64
        %705 = llvm.load %704 : !llvm.ptr -> f64
        %706 = llvm.call @printf(%686, %705) : (!llvm.ptr<i8>, f64) -> i32
        %707 = llvm.add %698, %697  : i64
        llvm.br ^bb63(%707 : i64)
      ^bb65:  // pred: ^bb63
        %708 = llvm.call @printf(%689) : (!llvm.ptr<i8>) -> i32
        %709 = llvm.add %693, %692  : i64
        llvm.br ^bb61(%709 : i64)
      ^bb66:  // pred: ^bb61
        %710 = llvm.mlir.addressof @int_frmt_spec : !llvm.ptr<array<4 x i8>>
        %711 = llvm.mlir.constant(0 : index) : i64
        %712 = llvm.getelementptr %710[%711, %711] : (!llvm.ptr<array<4 x i8>>, i64, i64) -> !llvm.ptr<i8>
        %713 = llvm.mlir.addressof @float_frmt_spec : !llvm.ptr<array<4 x i8>>
        %714 = llvm.mlir.constant(0 : index) : i64
        %715 = llvm.getelementptr %713[%714, %714] : (!llvm.ptr<array<4 x i8>>, i64, i64) -> !llvm.ptr<i8>
        %716 = llvm.mlir.addressof @nl : !llvm.ptr<array<2 x i8>>
        %717 = llvm.mlir.constant(0 : index) : i64
        %718 = llvm.getelementptr %716[%717, %717] : (!llvm.ptr<array<2 x i8>>, i64, i64) -> !llvm.ptr<i8>
        %719 = llvm.mlir.constant(0 : index) : i64
        %720 = llvm.mlir.constant(2 : index) : i64
        %721 = llvm.mlir.constant(1 : index) : i64
        llvm.br ^bb67(%719 : i64)
      ^bb67(%722: i64):  // 2 preds: ^bb66, ^bb71
        %723 = llvm.icmp "slt" %722, %720 : i64
        llvm.cond_br %723, ^bb68, ^bb72
      ^bb68:  // pred: ^bb67
        %724 = llvm.mlir.constant(0 : index) : i64
        %725 = llvm.mlir.constant(3 : index) : i64
        %726 = llvm.mlir.constant(1 : index) : i64
        llvm.br ^bb69(%724 : i64)
      ^bb69(%727: i64):  // 2 preds: ^bb68, ^bb70
        %728 = llvm.icmp "slt" %727, %725 : i64
        llvm.cond_br %728, ^bb70, ^bb71
      ^bb70:  // pred: ^bb69
        %729 = llvm.extractvalue %57[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %730 = llvm.mlir.constant(3 : index) : i64
        %731 = llvm.mul %722, %730  : i64
        %732 = llvm.add %731, %727  : i64
        %733 = llvm.getelementptr %729[%732] : (!llvm.ptr, i64) -> !llvm.ptr, f64
        %734 = llvm.load %733 : !llvm.ptr -> f64
        %735 = llvm.call @printf(%715, %734) : (!llvm.ptr<i8>, f64) -> i32
        %736 = llvm.add %727, %726  : i64
        llvm.br ^bb69(%736 : i64)
      ^bb71:  // pred: ^bb69
        %737 = llvm.call @printf(%718) : (!llvm.ptr<i8>) -> i32
        %738 = llvm.add %722, %721  : i64
        llvm.br ^bb67(%738 : i64)
      ^bb72:  // pred: ^bb67
        %739 = llvm.mlir.addressof @int_frmt_spec : !llvm.ptr<array<4 x i8>>
        %740 = llvm.mlir.constant(0 : index) : i64
        %741 = llvm.getelementptr %739[%740, %740] : (!llvm.ptr<array<4 x i8>>, i64, i64) -> !llvm.ptr<i8>
        %742 = llvm.mlir.addressof @float_frmt_spec : !llvm.ptr<array<4 x i8>>
        %743 = llvm.mlir.constant(0 : index) : i64
        %744 = llvm.getelementptr %742[%743, %743] : (!llvm.ptr<array<4 x i8>>, i64, i64) -> !llvm.ptr<i8>
        %745 = llvm.mlir.addressof @nl : !llvm.ptr<array<2 x i8>>
        %746 = llvm.mlir.constant(0 : index) : i64
        %747 = llvm.getelementptr %745[%746, %746] : (!llvm.ptr<array<2 x i8>>, i64, i64) -> !llvm.ptr<i8>
        %748 = llvm.mlir.constant(0 : index) : i64
        %749 = llvm.mlir.constant(2 : index) : i64
        %750 = llvm.mlir.constant(1 : index) : i64
        llvm.br ^bb73(%748 : i64)
      ^bb73(%751: i64):  // 2 preds: ^bb72, ^bb77
        %752 = llvm.icmp "slt" %751, %749 : i64
        llvm.cond_br %752, ^bb74, ^bb78
      ^bb74:  // pred: ^bb73
        %753 = llvm.mlir.constant(0 : index) : i64
        %754 = llvm.mlir.constant(3 : index) : i64
        %755 = llvm.mlir.constant(1 : index) : i64
        llvm.br ^bb75(%753 : i64)
      ^bb75(%756: i64):  // 2 preds: ^bb74, ^bb76
        %757 = llvm.icmp "slt" %756, %754 : i64
        llvm.cond_br %757, ^bb76, ^bb77
      ^bb76:  // pred: ^bb75
        %758 = llvm.extractvalue %40[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %759 = llvm.mlir.constant(3 : index) : i64
        %760 = llvm.mul %751, %759  : i64
        %761 = llvm.add %760, %756  : i64
        %762 = llvm.getelementptr %758[%761] : (!llvm.ptr, i64) -> !llvm.ptr, f64
        %763 = llvm.load %762 : !llvm.ptr -> f64
        %764 = llvm.call @printf(%744, %763) : (!llvm.ptr<i8>, f64) -> i32
        %765 = llvm.add %756, %755  : i64
        llvm.br ^bb75(%765 : i64)
      ^bb77:  // pred: ^bb75
        %766 = llvm.call @printf(%747) : (!llvm.ptr<i8>) -> i32
        %767 = llvm.add %751, %750  : i64
        llvm.br ^bb73(%767 : i64)
      ^bb78:  // pred: ^bb73
        %768 = llvm.extractvalue %210[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        llvm.call @free(%768) : (!llvm.ptr) -> ()
        %769 = llvm.extractvalue %193[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        llvm.call @free(%769) : (!llvm.ptr) -> ()
        %770 = llvm.extractvalue %176[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        llvm.call @free(%770) : (!llvm.ptr) -> ()
        %771 = llvm.extractvalue %159[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        llvm.call @free(%771) : (!llvm.ptr) -> ()
        %772 = llvm.extractvalue %142[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        llvm.call @free(%772) : (!llvm.ptr) -> ()
        %773 = llvm.extractvalue %125[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        llvm.call @free(%773) : (!llvm.ptr) -> ()
        %774 = llvm.extractvalue %108[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        llvm.call @free(%774) : (!llvm.ptr) -> ()
        %775 = llvm.extractvalue %91[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        llvm.call @free(%775) : (!llvm.ptr) -> ()
        %776 = llvm.extractvalue %74[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        llvm.call @free(%776) : (!llvm.ptr) -> ()
        %777 = llvm.extractvalue %57[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        llvm.call @free(%777) : (!llvm.ptr) -> ()
        %778 = llvm.extractvalue %40[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        llvm.call @free(%778) : (!llvm.ptr) -> ()
        llvm.return
      }
    }

    ```

8. 执行`./build/bin/toyc-ch6 -emit=llvm ./toy_mod/src/Ch6/codegen.toy -opt`重新生成LLVM IR

    ```llvm
    ; ModuleID = 'LLVMDialectModule'
    source_filename = "LLVMDialectModule"
    target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
    target triple = "x86_64-unknown-linux-gnu"

    @float_frmt_spec = internal constant [4 x i8] c"%f \00"
    @int_frmt_spec = internal constant [4 x i8] c"%d \00"

    ; Function Attrs: mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite)
    declare noalias noundef ptr @malloc(i64 noundef) local_unnamed_addr #0

    ; Function Attrs: mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite)
    declare void @free(ptr allocptr nocapture noundef) local_unnamed_addr #1

    ; Function Attrs: nofree nounwind
    declare noundef i32 @printf(ptr nocapture noundef readonly, ...) local_unnamed_addr #2

    ; Function Attrs: nounwind
    define void @main() local_unnamed_addr #3 !dbg !3 {
    .preheader42:
      %0 = tail call dereferenceable_or_null(48) ptr @malloc(i64 48), !dbg !6
      %1 = tail call dereferenceable_or_null(48) ptr @malloc(i64 48), !dbg !7
      store double 1.100000e+00, ptr %1, align 8, !dbg !7
      %2 = getelementptr double, ptr %1, i64 1, !dbg !7
      store double 2.200000e+00, ptr %2, align 8, !dbg !7
      %3 = getelementptr double, ptr %1, i64 2, !dbg !7
      store double 3.300000e+00, ptr %3, align 8, !dbg !7
      %4 = getelementptr double, ptr %1, i64 3, !dbg !7
      store double 4.400000e+00, ptr %4, align 8, !dbg !7
      %5 = getelementptr double, ptr %1, i64 4, !dbg !7
      store double 5.500000e+00, ptr %5, align 8, !dbg !7
      %6 = getelementptr double, ptr %1, i64 5, !dbg !7
      store double 6.600000e+00, ptr %6, align 8, !dbg !7
      store double 1.000000e+00, ptr %0, align 8, !dbg !6
      %7 = getelementptr double, ptr %0, i64 1, !dbg !6
      store double 2.000000e+00, ptr %7, align 8, !dbg !6
      %8 = getelementptr double, ptr %0, i64 2, !dbg !6
      store double 3.000000e+00, ptr %8, align 8, !dbg !6
      %9 = getelementptr double, ptr %0, i64 3, !dbg !6
      store double 4.000000e+00, ptr %9, align 8, !dbg !6
      %10 = getelementptr double, ptr %0, i64 4, !dbg !6
      store double 5.000000e+00, ptr %10, align 8, !dbg !6
      %11 = getelementptr double, ptr %0, i64 5, !dbg !6
      store double 6.000000e+00, ptr %11, align 8, !dbg !6
      %12 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @float_frmt_spec, double 1.100000e+00), !dbg !8
      %13 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @float_frmt_spec, double 1.760000e+01), !dbg !8
      %putchar18 = tail call i32 @putchar(i32 10), !dbg !8
      %14 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @float_frmt_spec, double 4.400000e+00), !dbg !8
      %15 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @float_frmt_spec, double 2.750000e+01), !dbg !8
      %putchar18.1 = tail call i32 @putchar(i32 10), !dbg !8
      %16 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @float_frmt_spec, double 0x4023CCCCCCCCCCCC), !dbg !8
      %17 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @float_frmt_spec, double 0x4043CCCCCCCCCCCC), !dbg !8
      %putchar18.2 = tail call i32 @putchar(i32 10), !dbg !8
      %18 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @float_frmt_spec, double 1.100000e+00), !dbg !9
      %19 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @float_frmt_spec, double 1.760000e+01), !dbg !9
      %putchar17 = tail call i32 @putchar(i32 10), !dbg !9
      %20 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @float_frmt_spec, double 4.400000e+00), !dbg !9
      %21 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @float_frmt_spec, double 2.750000e+01), !dbg !9
      %putchar17.1 = tail call i32 @putchar(i32 10), !dbg !9
      %22 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @float_frmt_spec, double 0x4023CCCCCCCCCCCC), !dbg !9
      %23 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @float_frmt_spec, double 0x4043CCCCCCCCCCCC), !dbg !9
      %putchar17.2 = tail call i32 @putchar(i32 10), !dbg !9
      %24 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @int_frmt_spec, i64 15), !dbg !10
      %25 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @int_frmt_spec, i64 14), !dbg !10
      %26 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @int_frmt_spec, i64 15), !dbg !10
      %putchar16 = tail call i32 @putchar(i32 10), !dbg !10
      %27 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @int_frmt_spec, i64 26), !dbg !10
      %28 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @int_frmt_spec, i64 27), !dbg !10
      %29 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @int_frmt_spec, i64 30), !dbg !10
      %putchar16.1 = tail call i32 @putchar(i32 10), !dbg !10
      %30 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @int_frmt_spec, i64 5), !dbg !11
      %31 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @int_frmt_spec, i64 8), !dbg !11
      %32 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @int_frmt_spec, i64 9), !dbg !11
      %putchar15 = tail call i32 @putchar(i32 10), !dbg !11
      %33 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @int_frmt_spec, i64 0), !dbg !11
      %34 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @int_frmt_spec, i64 1), !dbg !11
      %35 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @int_frmt_spec, i64 0), !dbg !11
      %putchar15.1 = tail call i32 @putchar(i32 10), !dbg !11
      %36 = load double, ptr %1, align 8, !dbg !12
      %37 = load double, ptr %0, align 8, !dbg !12
      %38 = fadd double %36, %37, !dbg !12
      %39 = load double, ptr %2, align 8, !dbg !12
      %40 = load double, ptr %7, align 8, !dbg !12
      %41 = fadd double %39, %40, !dbg !12
      %42 = load double, ptr %3, align 8, !dbg !12
      %43 = load double, ptr %8, align 8, !dbg !12
      %44 = fadd double %42, %43, !dbg !12
      %45 = load double, ptr %4, align 8, !dbg !12
      %46 = load double, ptr %9, align 8, !dbg !12
      %47 = fadd double %45, %46, !dbg !12
      %48 = load double, ptr %5, align 8, !dbg !12
      %49 = load double, ptr %10, align 8, !dbg !12
      %50 = fadd double %48, %49, !dbg !12
      %51 = load double, ptr %6, align 8, !dbg !12
      %52 = load double, ptr %11, align 8, !dbg !12
      %53 = fadd double %51, %52, !dbg !12
      %54 = fsub double %36, %37, !dbg !13
      %55 = fsub double %39, %40, !dbg !13
      %56 = fsub double %42, %43, !dbg !13
      %57 = fsub double %45, %46, !dbg !13
      %58 = fsub double %48, %49, !dbg !13
      %59 = fsub double %51, %52, !dbg !13
      %60 = fmul double %36, %37, !dbg !14
      %61 = fmul double %39, %40, !dbg !14
      %62 = fmul double %42, %43, !dbg !14
      %63 = fmul double %45, %46, !dbg !14
      %64 = fmul double %48, %49, !dbg !14
      %65 = fmul double %51, %52, !dbg !14
      %66 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @float_frmt_spec, double %38), !dbg !15
      %67 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @float_frmt_spec, double %41), !dbg !15
      %68 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @float_frmt_spec, double %44), !dbg !15
      %putchar14 = tail call i32 @putchar(i32 10), !dbg !15
      %69 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @float_frmt_spec, double %47), !dbg !15
      %70 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @float_frmt_spec, double %50), !dbg !15
      %71 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @float_frmt_spec, double %53), !dbg !15
      %putchar14.1 = tail call i32 @putchar(i32 10), !dbg !15
      %72 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @float_frmt_spec, double %54), !dbg !16
      %73 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @float_frmt_spec, double %55), !dbg !16
      %74 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @float_frmt_spec, double %56), !dbg !16
      %putchar13 = tail call i32 @putchar(i32 10), !dbg !16
      %75 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @float_frmt_spec, double %57), !dbg !16
      %76 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @float_frmt_spec, double %58), !dbg !16
      %77 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @float_frmt_spec, double %59), !dbg !16
      %putchar13.1 = tail call i32 @putchar(i32 10), !dbg !16
      %78 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @float_frmt_spec, double %60), !dbg !17
      %79 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @float_frmt_spec, double %61), !dbg !17
      %80 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @float_frmt_spec, double %62), !dbg !17
      %putchar = tail call i32 @putchar(i32 10), !dbg !17
      %81 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @float_frmt_spec, double %63), !dbg !17
      %82 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @float_frmt_spec, double %64), !dbg !17
      %83 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @float_frmt_spec, double %65), !dbg !17
      %putchar.1 = tail call i32 @putchar(i32 10), !dbg !17
      tail call void @free(ptr %1), !dbg !7
      tail call void @free(ptr %0), !dbg !6
      ret void, !dbg !18
    }

    ; Function Attrs: nofree nounwind
    declare noundef i32 @putchar(i32 noundef) local_unnamed_addr #2

    attributes #0 = { mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite) "alloc-family"="malloc" }
    attributes #1 = { mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite) "alloc-family"="malloc" }
    attributes #2 = { nofree nounwind }
    attributes #3 = { nounwind }

    !llvm.module.flags = !{!0}
    !llvm.dbg.cu = !{!1}

    !0 = !{i32 2, !"Debug Info Version", i32 3}
    !1 = distinct !DICompileUnit(language: DW_LANG_C, file: !2, producer: "MLIR", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
    !2 = !DIFile(filename: "codegen.toy", directory: ".")
    !3 = distinct !DISubprogram(name: "main", linkageName: "main", scope: !2, file: !2, line: 8, type: !4, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !1)
    !4 = !DISubroutineType(cc: DW_CC_normal, types: !5)
    !5 = !{}
    !6 = !DILocation(line: 11, scope: !3)
    !7 = !DILocation(line: 9, scope: !3)
    !8 = !DILocation(line: 20, column: 3, scope: !3)
    !9 = !DILocation(line: 21, column: 3, scope: !3)
    !10 = !DILocation(line: 26, column: 3, scope: !3)
    !11 = !DILocation(line: 27, column: 3, scope: !3)
    !12 = !DILocation(line: 29, column: 15, scope: !3)
    !13 = !DILocation(line: 30, column: 15, scope: !3)
    !14 = !DILocation(line: 31, column: 15, scope: !3)
    !15 = !DILocation(line: 32, column: 3, scope: !3)
    !16 = !DILocation(line: 33, column: 3, scope: !3)
    !17 = !DILocation(line: 34, column: 3, scope: !3)
    !18 = !DILocation(line: 8, column: 1, scope: !3)


    ```

9.  再次执行`./build/bin/toyc-ch6 -emit=jit ./toy_mod/src/Ch6/codegen.toy -opt`命令，可以看到整数类型值已经能够正确的print。

    ```
    1.100000 17.600000 
    4.400000 27.500000 
    9.900000 39.600000 
    1.100000 17.600000 
    4.400000 27.500000 
    9.900000 39.600000 
    15 14 15 
    26 27 30 
    5 8 9 
    0 1 0 
    2.100000 4.200000 6.300000 
    8.400000 10.500000 12.600000 
    0.100000 0.200000 0.300000 
    0.400000 0.500000 0.600000 
    1.100000 4.400000 9.900000 
    17.600000 27.500000 39.600000
    ```
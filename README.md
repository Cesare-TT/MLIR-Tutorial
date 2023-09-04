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

    ```MLIR
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

#### 将运算新增的BinOp lowering到Affine Dialect

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

    此时执行`./build/bin/toyc-ch6 -emit=mlir-affine ./toy_mod/src/Ch6/codegen.toy`命令生成lowering到affine dialect的mlir时，会报如下错误：

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

  
  

1. 
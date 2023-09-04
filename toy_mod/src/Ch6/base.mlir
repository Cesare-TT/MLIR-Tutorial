module {
  toy.func @main() {
    %0 = toy.constantint dense<[[7, 8, 9], [10, 11, 12]]> : tensor<2x3xi64>
    %1 = toy.constantint dense<[[13, 14, 15], [16, 17, 18]]> : tensor<2x3xi64>
    %2 = toy.or %0, %1 : tensor<2x3xi64>
    %3 = toy.and %0, %1 : tensor<2x3xi64>
    toy.print %2 : tensor<2x3xi64>
    toy.print %3 : tensor<2x3xi64>
    toy.return
  }
}

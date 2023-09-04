module {
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    // %11= mlir.convert %0 : tensor<2x3xf64> to tensor<2x3xi64>
    %1 = toy.constantint dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi64>
    %2 = toy.constantint dense<[1, 2, 3, 4, 5, 6]> : tensor<6xi64>
    %3 = toy.reshape(%2 : tensor<6xi64>) to tensor<2x3xi64>
    %4 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
    %5 = toy.mul %4, %4 : tensor<3x2xf64>
    toy.print %5 : tensor<3x2xf64>
    %6 = toy.or %1, %3 : tensor<2x3xi64>
    %7 = toy.and %1, %3 : tensor<2x3xi64>
    %8 = toy.add %0, %0 : tensor<2x3xf64>
    %9 = toy.sub %0, %0 : tensor<2x3xf64>
    %10 = toy.mul %0, %0 : tensor<2x3xf64>
    toy.print %6 : tensor<2x3xi64>
    toy.print %7 : tensor<2x3xi64>
    toy.print %8 : tensor<2x3xf64>
    toy.print %9 : tensor<2x3xf64>
    toy.print %10 : tensor<2x3xf64>
    toy.return
  }
}

// module {
//   toy.func private @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>) -> tensor<*xf64> {
//     %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
//     %1 = toy.transpose(%arg1 : tensor<*xf64>) to tensor<*xf64>
//     %2 = toy.mul %0, %1 : tensor<*xf64>
//     toy.return %2 : tensor<*xf64>
//   }
//   toy.func @main() {
//     %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
//     %1 = toy.reshape(%0 : tensor<2x3xf64>) to tensor<2x3xf64>
//     %2 = toy.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64>
//     %3 = toy.reshape(%2 : tensor<6xf64>) to tensor<2x3xf64>
//     %4 = toy.constantint dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi64>
//     %14= toy.reshape(%4 : tensor<2x3xi64>) to tensor<2x3xi64>
//     %5 = toy.constantint dense<[1, 2, 3, 4, 5, 6]> : tensor<6xi64>
//     %15= toy.reshape(%5 : tensor<6xi64>) to tensor<2x3xi64>
//     %6 = toy.generic_call @multiply_transpose(%1, %3) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
//     %7 = toy.generic_call @multiply_transpose(%3, %1) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
//     toy.print %7 : tensor<*xf64>
//     %8 = toy.or %14, %15 : (tensor<2x3xi64>, tensor<2x3xi64>) -> tensor<*xi64>
//     %9 = toy.and %14, %15 : (tensor<2x3xi64>, tensor<2x3xi64>) -> tensor<*xi64>
//     %10 = toy.add %1, %3 : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
//     %11 = toy.sub %1, %3 : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
//     %12 = toy.mul %1, %3 : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
//     toy.print %8 : tensor<*xi64>
//     toy.print %9 : tensor<*xi64>
//     toy.print %10 : tensor<*xf64>
//     toy.print %11 : tensor<*xf64>
//     toy.print %12 : tensor<*xf64>
//     toy.return
//   }
// }
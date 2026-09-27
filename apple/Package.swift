// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "Logitly",
    platforms: [
        .iOS(.v18),
        .macOS(.v15),
    ],
    products: [
        .library(name: "Logitly", targets: ["Logitly"]),
    ],
    targets: [
        .binaryTarget(
            name: "LlamaFramework",
            url: "https://github.com/ggml-org/llama.cpp/releases/download/b11147/llama-b11147-xcframework.zip",
            checksum: "d04512c7973241323faff78154340064d16be1330bba7dd97dbdffcd12890e06"
        ),
        .target(
            name: "Logitly",
            dependencies: ["LlamaFramework"]
        ),
        .testTarget(
            name: "LogitlyTests",
            dependencies: ["Logitly"]
        ),
    ]
)

import Foundation

public struct ModelProfile: Sendable, Equatable {
    public let name: String
    public let assistantSuffix: String?
    public let choiceLabels: [String]
    public let noulLabels: [String]
    public let scoreLabels: [String]

    public init(
        name: String = "generic",
        assistantSuffix: String? = nil,
        choiceLabels: [String] = Array("ABCDEFGHIJKLMNOPQRST").map(String.init),
        noulLabels: [String] = ["Y", "N"],
        scoreLabels: [String] = (0..<10).map(String.init)
    ) {
        self.name = name
        self.assistantSuffix = assistantSuffix
        self.choiceLabels = choiceLabels
        self.noulLabels = noulLabels
        self.scoreLabels = scoreLabels
    }

    public static let generic = ModelProfile()
    public static let lfm25 = ModelProfile(name: "lfm25", assistantSuffix: "<|im_start|>assistant\n")
}

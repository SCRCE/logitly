import Foundation
import Logitly
import SwiftUI

@MainActor
final class AppModel: ObservableObject {
    static let modelFilename = "LFM2.5-1.2B-Instruct-QAD-Q4_0"

    @Published private(set) var model: DecisionModel?
    @Published private(set) var status = "Not loaded"
    @Published private(set) var isLoading = false
    @Published var lastError: String?

    func ensureLoaded() async throws -> DecisionModel {
        if let model { return model }
        guard !isLoading else {
            while isLoading { try await Task.sleep(for: .milliseconds(50)) }
            if let model { return model }
            throw LogitlyError.modelLoadFailed
        }
        guard let url = Bundle.main.url(
            forResource: Self.modelFilename,
            withExtension: "gguf",
            subdirectory: "Models"
        ) ?? Bundle.main.url(forResource: Self.modelFilename, withExtension: "gguf") else {
            throw LogitlyError.modelNotFound("Bundled \(Self.modelFilename).gguf")
        }
        isLoading = true
        status = "Loading 696 MB model into Metal…"
        defer { isLoading = false }
        do {
            let loaded = try await DecisionModel.load(modelURL: url, profile: .lfm25)
            model = loaded
            status = "Ready · Metal · zero generation"
            lastError = nil
            return loaded
        } catch {
            status = "Load failed"
            lastError = error.localizedDescription
            throw error
        }
    }

    func unload() async {
        await model?.unload()
        model = nil
        status = "Not loaded"
    }
}

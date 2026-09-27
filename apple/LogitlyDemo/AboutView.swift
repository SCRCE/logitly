import SwiftUI

struct AboutView: View {
    @EnvironmentObject private var application: AppModel

    var body: some View {
        NavigationStack {
            List {
                Section("Execution contract") {
                    LabeledContent("Generated tokens", value: "0")
                    LabeledContent("Inference", value: "one prefill")
                    LabeledContent("Output", value: "restricted logits")
                    LabeledContent("Backend", value: "llama.cpp Metal")
                }
                Section("Pinned runtime") {
                    Text("llama.cpp b11147\nfee39dd92673ba0c08c8da96040ce53368b35188")
                        .font(.caption.monospaced()).textSelection(.enabled)
                    Text("LiquidAI LFM2.5-1.2B-Instruct QAD Q4_0\n8ed288026e23958ad9dfa92d53ed773a8eee7125")
                        .font(.caption.monospaced()).textSelection(.enabled)
                }
                Section("Memory") {
                    Text("Apple GPUs use unified memory. The model's approximately 696 MB weight mapping is included in the app's physical-memory footprint even while Metal executes it.")
                    LabeledContent("Current footprint", value: ByteCountFormatter.string(fromByteCount: Int64(AppMemory.physicalFootprint()), countStyle: .memory))
                }
                Section("Licenses") {
                    NavigationLink("LFM Open License v1.0") { LicenseView(resource: "LFM_LICENSE") }
                    NavigationLink("llama.cpp MIT License") { LicenseView(resource: "LLAMA_CPP_LICENSE") }
                }
                Section {
                    Button("Unload model", role: .destructive) { Task { await application.unload() } }
                        .disabled(application.model == nil)
                }
            }
            .navigationTitle("About")
        }
    }
}

struct LicenseView: View {
    let resource: String
    var body: some View {
        ScrollView {
            Text(contents).font(.caption.monospaced()).textSelection(.enabled).padding()
        }
        .navigationTitle("License")
    }

    private var contents: String {
        guard let url = Bundle.main.url(forResource: resource, withExtension: "txt", subdirectory: "ThirdParty")
                ?? Bundle.main.url(forResource: resource, withExtension: "txt"),
              let text = try? String(contentsOf: url, encoding: .utf8) else {
            return "License resource unavailable."
        }
        return text
    }
}

import SwiftUI

struct ProbabilityRow: View {
    let label: String
    let value: Double
    let selected: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack {
                Text(label).fontWeight(selected ? .bold : .regular)
                Spacer()
                Text(value, format: .percent.precision(.fractionLength(2))).monospacedDigit()
            }
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    Capsule().fill(.secondary.opacity(0.15))
                    Capsule().fill(selected ? Color.green : Color.teal)
                        .frame(width: geometry.size.width * max(0, min(1, value)))
                }
            }
            .frame(height: 8)
        }
        .accessibilityElement(children: .combine)
    }
}

struct StatusBanner: View {
    @EnvironmentObject private var application: AppModel

    var body: some View {
        HStack(spacing: 8) {
            if application.isLoading { ProgressView().controlSize(.small) }
            Circle().fill(application.model == nil ? .orange : .green).frame(width: 8, height: 8)
            Text(application.status).font(.caption).foregroundStyle(.secondary)
        }
    }
}

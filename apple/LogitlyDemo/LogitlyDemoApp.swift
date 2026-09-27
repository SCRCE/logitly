import SwiftUI
import UIKit

@main
struct LogitlyDemoApp: App {
    @StateObject private var application = AppModel()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(application)
        }
    }
}

struct RootView: View {
    @EnvironmentObject private var application: AppModel

    var body: some View {
        TabView {
            PlaygroundView()
                .tabItem { Label("Playground", systemImage: "slider.horizontal.3") }
            SnakeView()
                .tabItem { Label("Snake", systemImage: "square.grid.3x3.fill") }
            BenchmarkView()
                .tabItem { Label("Benchmark", systemImage: "gauge.with.dots.needle.67percent") }
            AboutView()
                .tabItem { Label("About", systemImage: "info.circle") }
        }
        .tint(.green)
        .onReceive(NotificationCenter.default.publisher(for: UIApplication.didReceiveMemoryWarningNotification)) { _ in
            Task { await application.unload() }
        }
    }
}

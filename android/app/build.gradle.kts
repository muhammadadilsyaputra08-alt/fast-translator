plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.fasttranslator"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.fasttranslator"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "1.0.0"

        externalNativeBuild {
            cmake {
                cppFlags += "-std=c++17"
                // Backend defaults to a generic CPU-only build in CI, since
                // this runner has no QNN SDK or Vulkan drivers installed.
                // Real QNN/Vulkan builds happen on a machine with those SDKs.
                arguments += listOf("-DENABLE_MEDUSA=ON")
            }
        }

        ndk {
            abiFilters += listOf("arm64-v8a")
        }
    }

    externalNativeBuild {
        cmake {
            path("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.activity:activity-ktx:1.9.3")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")
}

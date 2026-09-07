plugins {
    alias(libs.plugins.android.application)
}

// 1. Create a custom configuration to extract the AAR
val extractForNativeBuild by configurations.creating

android {
    namespace = "com.example.stablevqa"
    compileSdk = 36

    ndkVersion = "21.4.7075529"
    androidResources {
        // Prevent Android from compressing our ONNX graphs
        noCompress.add("onnx")
    }
    defaultConfig {
        applicationId = "com.example.stablevqa"
        minSdk = 24
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        externalNativeBuild {
            cmake {
                cppFlags += "-std=c++17 -frtti -fexceptions"
                arguments += "-DANDROID_STL=c++_shared"
                // Pass the extracted ONNX directory to CMake
                arguments += "-DONNXRUNTIME_DIR=${layout.buildDirectory.get().asFile.absolutePath}/onnxruntime_native"
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
    buildFeatures {
        viewBinding = true
        // Prefab is removed to prevent the CXX1429 crash
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    implementation(libs.androidx.constraintlayout)
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)

    // DocumentFile for safe folder access and CSV saving
    implementation("androidx.documentfile:documentfile:1.0.1")

    // ONNX Runtime implementation and extraction
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.19.2")
    extractForNativeBuild("com.microsoft.onnxruntime:onnxruntime-android:1.19.2")
}

// 2. The extraction task: Unzips the ONNX package before C++ compiles
tasks.register<Copy>("extractAARForNativeBuild") {
    from(configurations.getByName("extractForNativeBuild").map { zipTree(it) })
    into(layout.buildDirectory.dir("onnxruntime_native"))
    include("headers/**")
    include("jni/**")
}

// 3. Force the extraction to happen before CMake builds
tasks.withType<com.android.build.gradle.tasks.ExternalNativeBuildTask>().configureEach {
    dependsOn("extractAARForNativeBuild")
}
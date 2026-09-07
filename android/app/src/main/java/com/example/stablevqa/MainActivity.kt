package com.example.stablevqa

import android.content.ContentValues
import android.content.pm.PackageManager
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.drawable.BitmapDrawable
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.BatteryManager
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.documentfile.provider.DocumentFile
import com.example.stablevqa.databinding.ActivityMainBinding
import kotlinx.coroutines.*
import java.io.OutputStreamWriter
import java.util.Locale

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private var modelContextPtr: Long = 0

    private val modelInputWidth  = 224
    private val modelInputHeight = 224
    private val targetFrameCount = 32

    private var currentCsvUri: Uri? = null
    private var currentCsvFile: java.io.File? = null

    private fun getLiveCurrentUa(): Int {
        val bm      = getSystemService(BATTERY_SERVICE) as BatteryManager
        val current = bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)
        return kotlin.math.abs(current)
    }

    private fun getVmRssKb(): Int {
        try {
            val file = java.io.File("/proc/self/status")
            if (file.exists()) {
                file.useLines { lines ->
                    for (line in lines) {
                        if (line.startsWith("VmRSS:")) {
                            val parts = line.trim().split("\\s+".toRegex())
                            if (parts.size >= 2) return parts[1].toInt()
                        }
                    }
                }
            }
        } catch (e: Exception) {
            println("Error reading VmRSS: ${e.message}")
        }
        return 0
    }

    data class ClipMetrics(
        val score            : Float,
        val latencyS         : Float,
        val msDecode         : Float,
        val msPreprocess     : Float,
        val msBackbone       : Float,
        val msDeblur         : Float,
        val msFlow           : Float,
        val msMotion         : Float,
        val msQuality        : Float,
        val rssStart         : Float,
        val rssAfterBack     : Float,
        val rssAfterDeblur   : Float,
        val rssAfterFlow     : Float,
        val rssAfterMotion   : Float,
        val batteryMwh       : Float,

        val spatialIndicator : Float,
        val blurIndicator    : Float,
        val motionIndicator  : Float
    )

    data class VideoResult(
        val videoName    : String,
        val totalLatency : Float,
        val meanScore    : Float,
        val stdDev       : Float,
        val clips        : List<ClipMetrics>
    )

    private val pickFolderLauncher = registerForActivityResult(
        ActivityResultContracts.OpenDocumentTree()
    ) { treeUri: Uri? ->
        if (treeUri != null) {
            binding.statusText.text = getString(R.string.folder_selected)
            currentCsvUri  = null
            currentCsvFile = null
            runBatchPipeline(treeUri)
        } else {
            binding.statusText.text = getString(R.string.no_folder_selected)
        }
    }

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) pickFolderLauncher.launch(null)
        else binding.statusText.text = getString(R.string.storage_permission_denied)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.statusText.text = getString(R.string.loading_models)
        binding.selectFolderButton.isEnabled = false

        CoroutineScope(Dispatchers.IO).launch {
            modelContextPtr = initStableVQAModels(assets, cacheDir.absolutePath)
            withContext(Dispatchers.Main) {
                if (modelContextPtr != 0L) {
                    binding.statusText.text = getString(R.string.models_ready)
                    binding.selectFolderButton.isEnabled = true
                } else {
                    binding.statusText.text = getString(R.string.models_failed)
                }
            }
        }

        binding.selectFolderButton.setOnClickListener {
            if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q &&
                ContextCompat.checkSelfPermission(
                    this, android.Manifest.permission.READ_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED) {
                requestPermissionLauncher.launch(
                    android.Manifest.permission.READ_EXTERNAL_STORAGE)
            } else {
                pickFolderLauncher.launch(null)
            }
        }
    }

    private fun runBatchPipeline(treeUri: Uri) {
        CoroutineScope(Dispatchers.IO).launch {

            val docTree         = DocumentFile.fromTreeUri(this@MainActivity, treeUri)
            val validExtensions = listOf(".mp4", ".avi", ".mov", ".mkv", ".webm")

            val videoFiles = docTree
                ?.listFiles()
                ?.filter { file ->
                    file.isFile && validExtensions.any { ext ->
                        file.name?.lowercase()?.endsWith(ext) == true
                    }
                }
                ?.sortedBy { it.name }
                ?: emptyList()

            if (videoFiles.isEmpty()) {
                withContext(Dispatchers.Main) {
                    binding.statusText.text = getString(R.string.no_videos_found)
                }
                return@launch
            }

            withContext(Dispatchers.Main) {
                binding.statusText.text = getString(R.string.found_videos, videoFiles.size)
            }

            val allResults      = mutableListOf<VideoResult>()
            val imageArea       = modelInputWidth * modelInputHeight
            val normFramesArray = FloatArray(targetFrameCount * 3 * imageArea)
            val rawFramesArray  = FloatArray(targetFrameCount * 3 * imageArea)
            val tempPixels      = IntArray(imageArea)

            val meanR = 123.675f; val stdR = 58.395f
            val meanG = 116.28f;  val stdG = 57.12f
            val meanB = 103.53f;  val stdB = 57.375f

            for ((vidIdx, docFile) in videoFiles.withIndex()) {
                val videoName = docFile.name ?: "video_$vidIdx"
                val videoUri  = docFile.uri

                withContext(Dispatchers.Main) {
                    binding.statusText.text =
                        getString(R.string.processing_video, vidIdx + 1, videoFiles.size, videoName)
                }
                println("Processing: $videoName")

                var totalFrames = 0
                var durationMs  = 0L
                val metaR = MediaMetadataRetriever()
                try {
                    metaR.setDataSource(this@MainActivity, videoUri)
                    durationMs = metaR
                        .extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                        ?.toLongOrNull() ?: 0L
                    val fc = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P)
                        metaR.extractMetadata(
                            MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT)
                            ?.toIntOrNull() ?: 0
                    else 0
                    totalFrames = if (fc > 0) fc else (durationMs / 1000.0 * 30.0).toInt()
                } catch (e: Exception) {
                    println("  Metadata error for $videoName: ${e.message}"); continue
                } finally { metaR.release() }

                if (totalFrames == 0) {
                    println("  Skipping $videoName — 0 frames detected.")
                    continue
                }

                val numClips        = 4
                val oriClipLen      = 64
                val maxStart        = maxOf(0, totalFrames - oriClipLen)
                val clipLabels      = arrayOf("12.5%", "37.5%", "62.5%", "87.5%")
                val clipMetricsList = mutableListOf<ClipMetrics>()
                var totalLatency    = 0f

                for (segment in 0 until numClips) {
                    val pos        = (segment + 0.5f) / numClips.toFloat()
                    val startFrame = (pos * maxStart).toInt()

                    val baselineRssKb = getVmRssKb()
                    println("  Clip ${segment + 1} baseline RSS: $baselineRssKb KB")

                    var collected         = 0
                    var msDecodeTotal     = 0L
                    var msPreprocessTotal = 0L

                    val retriever = MediaMetadataRetriever()
                    try {
                        retriever.setDataSource(this@MainActivity, videoUri)

                        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                            val frameRequestCount = maxOf(0, minOf(oriClipLen, totalFrames - startFrame))
                            for (i in 0 until frameRequestCount) {
                                if (collected >= targetFrameCount) break
                                if (i % 2 != 0) {
                                    continue
                                }

                                var t0 = System.currentTimeMillis()
                                val raw = retriever.getFrameAtIndex(startFrame + i) ?: continue
                                
                                msDecodeTotal += (System.currentTimeMillis() - t0)

                                t0 = System.currentTimeMillis()
                                val bmp = Bitmap.createScaledBitmap(
                                    raw, modelInputWidth, modelInputHeight, true)
                                if (raw != bmp) raw.recycle()
                                msDecodeTotal += (System.currentTimeMillis() - t0)

                                if (vidIdx == 0 && segment == 0 && collected == 0) {
                                    val ui = bmp.copy(Bitmap.Config.ARGB_8888, false)
                                    withContext(Dispatchers.Main) {
                                        val old = binding.flowVisualization.drawable
                                        if (old is BitmapDrawable) old.bitmap?.recycle()
                                        binding.flowVisualization.setImageBitmap(ui)
                                    }
                                }

                                val t1 = System.currentTimeMillis()
                                bmp.getPixels(tempPixels, 0, modelInputWidth,
                                    0, 0, modelInputWidth, modelInputHeight)
                                bmp.recycle()

                                val fo   = collected * 3 * imageArea
                                var rOff = fo
                                var gOff = fo + imageArea
                                var bOff = fo + 2 * imageArea
                                for (p in 0 until imageArea) {
                                    val c = tempPixels[p]
                                    val r = (c shr 16 and 0xFF).toFloat()
                                    val g = (c shr 8  and 0xFF).toFloat()
                                    val b = (c        and 0xFF).toFloat()
                                    normFramesArray[rOff] = (r - meanR) / stdR
                                    normFramesArray[gOff] = (g - meanG) / stdG
                                    normFramesArray[bOff] = (b - meanB) / stdB
                                    rawFramesArray[rOff]  = r
                                    rawFramesArray[gOff]  = g
                                    rawFramesArray[bOff]  = b
                                    rOff++; gOff++; bOff++
                                }
                                msPreprocessTotal += (System.currentTimeMillis() - t1)
                                collected++
                            }
                        } else {
                            val frameTimeUs = if (totalFrames > 0 && durationMs > 0L)
                                (durationMs * 1000L) / totalFrames else 33_333L
                            for (i in 0 until oriClipLen) {
                                if (collected >= targetFrameCount) break
                                if (i % 2 != 0) continue
                                val fi = startFrame + i
                                if (fi >= totalFrames) break

                                val t0  = System.currentTimeMillis()
                                val raw = retriever.getFrameAtTime(
                                    fi * frameTimeUs,
                                    MediaMetadataRetriever.OPTION_CLOSEST) ?: break
                                val bmp = if (raw.width != modelInputWidth ||
                                    raw.height != modelInputHeight) {
                                    val s = Bitmap.createScaledBitmap(
                                        raw, modelInputWidth, modelInputHeight, true)
                                    raw.recycle(); s
                                } else raw
                                msDecodeTotal += (System.currentTimeMillis() - t0)

                                val t1 = System.currentTimeMillis()
                                bmp.getPixels(tempPixels, 0, modelInputWidth,
                                    0, 0, modelInputWidth, modelInputHeight)
                                bmp.recycle()

                                val fo   = collected * 3 * imageArea
                                var rOff = fo
                                var gOff = fo + imageArea
                                var bOff = fo + 2 * imageArea
                                for (p in 0 until imageArea) {
                                    val col = tempPixels[p]
                                    val r   = (col shr 16 and 0xFF).toFloat()
                                    val g   = (col shr 8  and 0xFF).toFloat()
                                    val b   = (col        and 0xFF).toFloat()
                                    normFramesArray[rOff] = (r - meanR) / stdR
                                    normFramesArray[gOff] = (g - meanG) / stdG
                                    normFramesArray[bOff] = (b - meanB) / stdB
                                    rawFramesArray[rOff]  = r
                                    rawFramesArray[gOff]  = g
                                    rawFramesArray[bOff]  = b
                                    rOff++; gOff++; bOff++
                                }
                                msPreprocessTotal += (System.currentTimeMillis() - t1)
                                collected++
                            }
                        }
                    } catch (e: Exception) {
                        println("  Clip ${segment + 1} frame extraction error: ${e.message}")
                    } finally { retriever.release() }

                    if (collected in 1 until targetFrameCount) {
                        val tPad = System.currentTimeMillis()
                        for (i in collected until targetFrameCount) {
                            val s = (i - 1) * 3 * imageArea
                            val d = i       * 3 * imageArea
                            System.arraycopy(normFramesArray, s, normFramesArray, d, 3 * imageArea)
                            System.arraycopy(rawFramesArray,  s, rawFramesArray,  d, 3 * imageArea)
                        }
                        collected = targetFrameCount
                        msPreprocessTotal += (System.currentTimeMillis() - tPad)
                    }

                    System.gc(); delay(50)

                    if (collected == targetFrameCount) {
                        var isInferencing = true
                        val currentReadings = mutableListOf<Int>()
                        val batteryPoller = CoroutineScope(Dispatchers.IO).launch {
                            while (isInferencing) {
                                currentReadings.add(getLiveCurrentUa())
                                delay(500)
                            }
                        }

                        val res = runStableVQABenchmark(
                            modelContextPtr, normFramesArray, rawFramesArray)

                        isInferencing = false
                        batteryPoller.join()

                        val avgCurrentUa   =
                            if (currentReadings.isNotEmpty()) currentReadings.average() else 0.0
                        val avgCurrentMa   = avgCurrentUa / 1000.0
                        val latencyHours   = if (res != null) res[1] / 3600.0 else 0.0
                        val nominalVoltage = 3.7
                        val batteryMwh     = (avgCurrentMa * nominalVoltage * latencyHours).toFloat()

                        if (res != null && res.size == 15) {
                            val mos   = res[0]
                            val label = when {
                                mos <= 20 -> "Very Unstable"
                                mos <= 40 -> "Unstable"
                                mos <= 60 -> "Moderately Stable"
                                mos <= 80 -> "Stable"
                                else      -> "Highly Stable"
                            }

                            val metrics = ClipMetrics(
                                score            = mos,
                                latencyS         = res[1],
                                msDecode         = msDecodeTotal.toFloat(),
                                msPreprocess     = msPreprocessTotal.toFloat(),
                                msBackbone       = res[2],
                                msDeblur         = res[3],
                                msFlow           = res[4],
                                msMotion         = res[5],
                                msQuality        = res[6],
                                rssStart         = res[7],
                                rssAfterBack     = res[8],
                                rssAfterDeblur   = res[9],
                                rssAfterFlow     = res[10],
                                rssAfterMotion   = res[11],
                                batteryMwh       = batteryMwh,
                                spatialIndicator = res[12],
                                blurIndicator    = res[13],
                                motionIndicator  = res[14]
                            )

                            clipMetricsList.add(metrics)
                            totalLatency += res[1]

                            println(
                                "  Clip ${segment + 1} [${clipLabels[segment]}] | " +
                                        "MOS=${String.format(Locale.US, "%.2f", mos)} | $label | " +
                                        "SpatialIndicator=${String.format(Locale.US, "%.1f", res[12])} | " +
                                        "BlurIndicator=${String.format(Locale.US, "%.1f", res[13])} | " +
                                        "MotionIndicator=${String.format(Locale.US, "%.1f", res[14])} | " +
                                        "time=${String.format(Locale.US, "%.2f", res[1])}s"
                            )
                        }

                        if (segment < numClips - 1) delay(2000)
                    }
                }

                if (clipMetricsList.isNotEmpty()) {
                    val scores   = clipMetricsList.map { it.score }
                    val mean     = scores.sum() / scores.size
                    var variance = 0f
                    scores.forEach { variance += (it - mean) * (it - mean) }
                    val std = kotlin.math.sqrt((variance / scores.size).toDouble()).toFloat()

                    allResults.add(VideoResult(
                        videoName    = videoName,
                        totalLatency = totalLatency,
                        meanScore    = mean,
                        stdDev       = std,
                        clips        = clipMetricsList.toList()
                    ))

                    val finalLabel = when {
                        mean <= 20 -> "Very Unstable"
                        mean <= 40 -> "Unstable"
                        mean <= 60 -> "Moderately Stable"
                        mean <= 80 -> "Stable"
                        else       -> "Highly Stable"
                    }

                    println(
                        "Finished $videoName — " +
                                "mean=${String.format(Locale.US, "%.2f", mean)} | " +
                                "$finalLabel | " +
                                "total=${String.format(Locale.US, "%.2f", totalLatency)}s"
                    )

                    writeCsv(allResults)
                }

                if (vidIdx < videoFiles.size - 1) {
                    withContext(Dispatchers.Main) {
                        binding.statusText.text =
                            getString(R.string.cooling_down, videoName, vidIdx + 1, videoFiles.size)
                    }
                    delay(3000)
                }
            } // end video loop

            withContext(Dispatchers.Main) {
                binding.statusText.text = if (allResults.isNotEmpty())
                    getString(R.string.results_saved, allResults.size)
                else
                    getString(R.string.no_results)
            }
        }
    }

    private fun writeCsv(results: List<VideoResult>) {
        try {
            val fileName   = "stablevqa_results.csv"
            val clipLabels = arrayOf("12.5%", "37.5%", "62.5%", "87.5%")

            val csv = buildString {
                appendLine(
                    "VideoName,Clip,StartPos,Score," +
                            "SpatialIndicator,BlurIndicator,MotionIndicator," +
                            "TotalLatency(s),FrameDecode(ms),Preprocess(ms)," +
                            "Backbone(ms),Deblur(ms),Flow(ms),Motion(ms),QualityHead(ms)," +
                            "RSS_Start(KB),RSS_AfterBackbone(KB),RSS_AfterDeblur(KB)," +
                            "RSS_AfterFlow(KB),RSS_AfterMotion(KB)," +
                            "BackboneRAMDelta(KB),DeblurRAMDelta(KB)," +
                            "FlowRAMDelta(KB),MotionRAMDelta(KB)," +
                            "BatteryUsed(mWh),VideoMeanScore,VideoStdDev,StabilityLabel"
                )
                for (r in results) {
                    r.clips.forEachIndexed { idx, c ->
                        val backDelta   = c.rssAfterBack   - c.rssStart
                        val deblurDelta = c.rssAfterDeblur - c.rssAfterBack
                        val flowDelta   = c.rssAfterFlow   - c.rssAfterDeblur
                        val motionDelta = c.rssAfterMotion - c.rssAfterFlow

                        val stabilityLabel = when {
                            r.meanScore <= 20 -> "Very Unstable"
                            r.meanScore <= 40 -> "Unstable"
                            r.meanScore <= 60 -> "Moderately Stable"
                            r.meanScore <= 80 -> "Stable"
                            else              -> "Highly Stable"
                        }

                        appendLine(
                            "${r.videoName},${idx + 1},${clipLabels.getOrElse(idx) { "?" }}," +
                                    "${String.format(Locale.US, "%.4f", c.score)}," +
                                    "${String.format(Locale.US, "%.2f", c.spatialIndicator)}," +
                                    "${String.format(Locale.US, "%.2f", c.blurIndicator)}," +
                                    "${String.format(Locale.US, "%.2f", c.motionIndicator)}," +
                                    "${String.format(Locale.US, "%.3f", c.latencyS)}," +
                                    "${String.format(Locale.US, "%.0f", c.msDecode)}," +
                                    "${String.format(Locale.US, "%.0f", c.msPreprocess)}," +
                                    "${String.format(Locale.US, "%.0f", c.msBackbone)}," +
                                    "${String.format(Locale.US, "%.0f", c.msDeblur)}," +
                                    "${String.format(Locale.US, "%.0f", c.msFlow)}," +
                                    "${String.format(Locale.US, "%.0f", c.msMotion)}," +
                                    "${String.format(Locale.US, "%.0f", c.msQuality)}," +
                                    "${c.rssStart.toInt()},${c.rssAfterBack.toInt()}," +
                                    "${c.rssAfterDeblur.toInt()},${c.rssAfterFlow.toInt()}," +
                                    "${c.rssAfterMotion.toInt()},${backDelta.toInt()}," +
                                    "${deblurDelta.toInt()},${flowDelta.toInt()},${motionDelta.toInt()}," +
                                    (if (c.batteryMwh < 0) "N/A"
                                    else String.format(Locale.US, "%.3f", c.batteryMwh)) +
                                    ",${String.format(Locale.US, "%.4f", r.meanScore)}," +
                                    "${String.format(Locale.US, "%.4f", r.stdDev)}," +
                                    stabilityLabel
                        )
                    }
                }
            }

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                if (currentCsvUri == null) {
                    val cv = ContentValues().apply {
                        put(MediaStore.Downloads.DISPLAY_NAME, fileName)
                        put(MediaStore.Downloads.MIME_TYPE, "text/csv")
                        put(MediaStore.Downloads.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS)
                    }
                    currentCsvUri = contentResolver.insert(
                        MediaStore.Downloads.EXTERNAL_CONTENT_URI, cv)
                }
                currentCsvUri?.let { uri ->
                    contentResolver.openOutputStream(uri, "wt")?.use { os ->
                        OutputStreamWriter(os).use { w -> w.write(csv) }
                    }
                }
            } else {
                if (currentCsvFile == null) {
                    val dir = Environment.getExternalStoragePublicDirectory(
                        Environment.DIRECTORY_DOWNLOADS)
                    dir.mkdirs()
                    currentCsvFile = java.io.File(dir, fileName)
                }
                currentCsvFile?.writeText(csv)
            }
        } catch (e: Exception) {
            println("CSV write failed: ${e.message}")
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        if (modelContextPtr != 0L) {
            destroyStableVQAModels(modelContextPtr)
            modelContextPtr = 0L
        }
    }

    external fun initStableVQAModels(assetManager: AssetManager, cachePath: String): Long
    external fun runStableVQABenchmark(
        contextPtr : Long,
        normFrames : FloatArray,
        rawFrames  : FloatArray
    ): FloatArray?
    external fun destroyStableVQAModels(contextPtr: Long)

    companion object {
        init { System.loadLibrary("video_stability_engine") }
    }
}
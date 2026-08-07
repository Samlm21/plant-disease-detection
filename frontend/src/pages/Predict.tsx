import React, { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Upload, 
  X, 
  FileImage, 
  Sparkles, 
  ShieldAlert, 
  Clock, 
  Download, 
  RefreshCw, 
  CheckCircle2, 
  AlertCircle,
  HelpCircle
} from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { ConfidenceBar } from '@/components/prediction/ConfidenceBar';
import { predictionService } from '@/services/predictionService';
import { historyService } from '@/services/historyService';
import { PredictionResponse } from '@/types/prediction';

export const Predict: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<'resnet18' | 'baseline'>('resnet18');
  const [enableGradCam, setEnableGradCam] = useState<boolean>(true);
  
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);

  // Drag and drop handlers
  const handleFileChange = (file: File) => {
    setError(null);
    if (!file.type.startsWith('image/')) {
      setError('Please upload a valid image file (JPEG, PNG, WebP).');
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      setError('File size exceeds 10MB limit.');
      return;
    }

    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResult(null);
  };

  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  }, []);

  const handleClear = () => {
    setSelectedFile(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
  };

  const handleSubmit = async () => {
    if (!selectedFile) return;

    setIsLoading(true);
    setError(null);

    try {
      const response = await predictionService.predict(
        selectedFile, 
        selectedModel, 
        enableGradCam
      );
      setResult(response);

      // Save to client history
      historyService.add({
        ...response,
        imageUrl: previewUrl || '',
        imageName: selectedFile.name
      });
    } catch (err: any) {
      setError(err.message || 'An unexpected error occurred during prediction.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto space-y-8 py-4">
      {/* Header */}
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <Badge variant="emerald" className="text-xs font-mono">Inference Dashboard</Badge>
        </div>
        <h1 className="text-3xl font-extrabold text-white tracking-tight">Plant Leaf Diagnosis</h1>
        <p className="text-gray-400 text-sm">
          Upload a high-resolution photo of an affected leaf to run AI disease identification.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column: Config & File Upload */}
        <div className="lg:col-span-5 space-y-6">
          {/* Controls */}
          <Card className="space-y-4">
            <h3 className="text-sm font-semibold text-gray-200">Inference Configuration</h3>
            
            <div className="space-y-2">
              <label className="text-xs text-gray-400">Select Architecture Model</label>
              <div className="grid grid-cols-2 gap-2">
                {(['resnet18', 'baseline'] as const).map((model) => (
                  <button
                    key={model}
                    type="button"
                    onClick={() => setSelectedModel(model)}
                    className={`py-2 px-3 rounded-lg text-xs font-medium border transition-all text-center ${
                      selectedModel === model
                        ? 'bg-emerald-500/10 border-emerald-500 text-emerald-400'
                        : 'bg-gray-900/40 border-gray-800 text-gray-400 hover:border-gray-700'
                    }`}
                  >
                    {model === 'resnet18' ? 'ResNet-18 (Recommended)' : 'Custom CNN Baseline'}
                  </button>
                ))}
              </div>
            </div>

            <div className="flex items-center justify-between pt-2">
              <label className="text-xs text-gray-300 flex items-center gap-1.5 cursor-pointer">
                <span>Generate Grad-CAM Heatmap</span>
              </label>
              <input
                type="checkbox"
                checked={enableGradCam}
                onChange={(e) => setEnableGradCam(e.target.checked)}
                className="rounded bg-gray-800 border-gray-700 text-emerald-500 focus:ring-emerald-500/20"
              />
            </div>
          </Card>

          {/* Upload Dropzone */}
          <Card className="space-y-4">
            {!previewUrl ? (
              <div
                onDragOver={(e) => e.preventDefault()}
                onDrop={handleDrop}
                className="border-2 border-dashed border-gray-800 hover:border-emerald-500/50 rounded-xl p-8 text-center space-y-4 transition-all bg-gray-950/20 cursor-pointer group"
              >
                <div className="w-12 h-12 rounded-full bg-emerald-500/10 text-emerald-400 flex items-center justify-center mx-auto group-hover:scale-110 transition-transform">
                  <Upload className="w-6 h-6" />
                </div>
                <div className="space-y-1">
                  <p className="text-sm font-medium text-gray-200">
                    Drag and drop leaf image, or <span className="text-emerald-400 hover:underline">browse</span>
                  </p>
                  <p className="text-xs text-gray-500">Supports JPG, PNG, WEBP (Max 10MB)</p>
                </div>
                <input
                  type="file"
                  accept="image/*"
                  onChange={(e) => e.target.files?.[0] && handleFileChange(e.target.files[0])}
                  className="hidden"
                  id="file-upload"
                />
                <label htmlFor="file-upload" className="inline-block">
                  <Button variant="secondary" size="sm" type="button" className="pointer-events-none">
                    Select Image File
                  </Button>
                </label>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="relative rounded-xl overflow-hidden border border-gray-800 bg-black/40 aspect-square flex items-center justify-center">
                  <img
                    src={previewUrl}
                    alt="Leaf preview"
                    className="max-h-full max-w-full object-contain"
                  />
                  <button
                    onClick={handleClear}
                    className="absolute top-3 right-3 p-1.5 rounded-full bg-gray-900/80 text-gray-400 hover:text-white border border-gray-700/80 transition-all"
                    title="Remove Image"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>

                <div className="flex items-center justify-between text-xs text-gray-400 px-1">
                  <span className="truncate max-w-[200px]">{selectedFile?.name}</span>
                  <span>{((selectedFile?.size || 0) / (1024 * 1024)).toFixed(2)} MB</span>
                </div>

                <Button
                  onClick={handleSubmit}
                  isLoading={isLoading}
                  className="w-full gap-2 py-3 text-sm font-semibold"
                >
                  <Sparkles className="w-4 h-4" /> Run Diagnosis
                </Button>
              </div>
            )}

            {error && (
              <div className="p-3 rounded-lg bg-rose-500/10 border border-rose-500/20 text-rose-400 text-xs flex items-center gap-2">
                <AlertCircle className="w-4 h-4 flex-shrink-0" />
                <span>{error}</span>
              </div>
            )}
          </Card>
        </div>

        {/* Right Column: Inference Results */}
        <div className="lg:col-span-7 space-y-6">
          <AnimatePresence mode="wait">
            {isLoading ? (
              <Card key="loading" className="p-12 text-center space-y-6 border-emerald-500/20">
                <div className="relative w-16 h-16 mx-auto">
                  <div className="absolute inset-0 rounded-full border-4 border-emerald-500/20 border-t-emerald-400 animate-spin" />
                  <Sparkles className="w-6 h-6 text-emerald-400 absolute inset-0 m-auto animate-pulse" />
                </div>
                <div className="space-y-2">
                  <h3 className="text-lg font-bold text-white">Running Model Inference...</h3>
                  <p className="text-xs text-gray-400 max-w-sm mx-auto">
                    Processing tensor tensors through {selectedModel === 'resnet18' ? 'ResNet-18' : 'Custom CNN'} and evaluating Grad-CAM gradients.
                  </p>
                </div>
              </Card>
            ) : result ? (
              <motion.div
                key="result"
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                className="space-y-6"
              >
                {/* Main Diagnosis Summary Card */}
                <Card className="space-y-6 border-emerald-500/30 bg-gradient-to-b from-gray-900/80 to-gray-950">
                  <div className="flex flex-wrap items-start justify-between gap-4 border-b border-gray-800 pb-4">
                    <div className="space-y-1">
                      <div className="flex items-center gap-2">
                        <Badge variant="emerald">{result.diseaseInfo.crop}</Badge>
                        <Badge 
                          variant={
                            result.diseaseInfo.severity === 'Critical' || result.diseaseInfo.severity === 'High' 
                              ? 'rose' 
                              : result.diseaseInfo.severity === 'Medium' 
                              ? 'amber' 
                              : 'emerald'
                          }
                        >
                          {result.diseaseInfo.severity} Severity
                        </Badge>
                      </div>
                      <h2 className="text-2xl font-bold text-white tracking-tight">{result.diseaseInfo.name}</h2>
                    </div>

                    <div className="text-right font-mono text-xs text-gray-400 space-y-1">
                      <div className="flex items-center justify-end gap-1 text-gray-300">
                        <Clock className="w-3.5 h-3.5" />
                        <span>{result.inferenceTimeMs}ms</span>
                      </div>
                      <p>Model: {result.diseaseInfo.category}</p>
                    </div>
                  </div>

                  {/* Confidence Bar */}
                  <ConfidenceBar 
                    label="Prediction Confidence" 
                    confidence={result.confidence} 
                  />

                  {/* Description */}
                  <div className="space-y-2">
                    <h4 className="text-xs font-semibold text-gray-300 uppercase tracking-wider">Diagnosis Overview</h4>
                    <p className="text-sm text-gray-400 leading-relaxed">{result.diseaseInfo.description}</p>
                  </div>

                  {/* Top-5 Predictions Probability Chart */}
                  <div className="space-y-3 pt-2">
                    <h4 className="text-xs font-semibold text-gray-300 uppercase tracking-wider">Top 5 Model Probabilities</h4>
                    <div className="space-y-2">
                      {result.topPredictions.map((pred, idx) => (
                        <div key={idx} className="space-y-1">
                          <div className="flex justify-between text-xs font-mono">
                            <span className="text-gray-300 truncate">{pred.className.replace(/___/g, ' - ')}</span>
                            <span className="text-emerald-400">{(pred.probability * 100).toFixed(1)}%</span>
                          </div>
                          <div className="w-full h-1.5 bg-gray-800 rounded-full overflow-hidden">
                            <div 
                              className="h-full bg-emerald-500 rounded-full transition-all duration-500"
                              style={{ width: `${pred.probability * 100}%` }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </Card>

                {/* Grad-CAM Heatmap Visualization (if available) */}
                {result.gradCamHeatmapUrl && (
                  <Card className="space-y-4">
                    <h3 className="text-sm font-semibold text-gray-200 flex items-center gap-2">
                      <Sparkles className="w-4 h-4 text-emerald-400" />
                      Grad-CAM Spatial Explainability Overlay
                    </h3>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <p className="text-xs text-gray-400">Input Image</p>
                        <div className="rounded-lg overflow-hidden border border-gray-800 aspect-square">
                          <img src={previewUrl!} alt="Input" className="w-full h-full object-cover" />
                        </div>
                      </div>
                      <div className="space-y-2">
                        <p className="text-xs text-gray-400">Gradient Activation Map</p>
                        <div className="rounded-lg overflow-hidden border border-emerald-500/30 aspect-square">
                          <img src={result.gradCamHeatmapUrl} alt="Heatmap" className="w-full h-full object-cover" />
                        </div>
                      </div>
                    </div>
                  </Card>
                )}

                {/* Treatment Recommendations Accordion */}
                <Card className="space-y-4">
                  <h3 className="text-sm font-semibold text-gray-200">Recommended Treatment Plan</h3>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="p-3 rounded-lg bg-emerald-500/5 border border-emerald-500/20 space-y-2">
                      <p className="text-xs font-bold text-emerald-400 uppercase">Organic Control</p>
                      <ul className="text-xs text-gray-400 space-y-1 list-disc list-inside">
                        {result.diseaseInfo.treatment.organic.map((item, i) => (
                          <li key={i}>{item}</li>
                        ))}
                      </ul>
                    </div>

                    <div className="p-3 rounded-lg bg-cyan-500/5 border border-cyan-500/20 space-y-2">
                      <p className="text-xs font-bold text-cyan-400 uppercase">Chemical Control</p>
                      <ul className="text-xs text-gray-400 space-y-1 list-disc list-inside">
                        {result.diseaseInfo.treatment.chemical.map((item, i) => (
                          <li key={i}>{item}</li>
                        ))}
                      </ul>
                    </div>

                    <div className="p-3 rounded-lg bg-amber-500/5 border border-amber-500/20 space-y-2">
                      <p className="text-xs font-bold text-amber-400 uppercase">Prevention</p>
                      <ul className="text-xs text-gray-400 space-y-1 list-disc list-inside">
                        {result.diseaseInfo.treatment.prevention.map((item, i) => (
                          <li key={i}>{item}</li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </Card>
              </motion.div>
            ) : (
              <Card key="empty" className="p-12 text-center space-y-4 border-dashed border-gray-800">
                <FileImage className="w-10 h-10 text-gray-600 mx-auto" />
                <div className="space-y-1">
                  <h3 className="text-base font-semibold text-gray-300">Awaiting Image Input</h3>
                  <p className="text-xs text-gray-500 max-w-sm mx-auto">
                    Select or drop a plant leaf image on the left panel to execute inference diagnosis.
                  </p>
                </div>
              </Card>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
};

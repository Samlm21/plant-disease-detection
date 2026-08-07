import React from 'react';
import { motion } from 'framer-motion';
import { Database, Cpu, CheckCircle2, Layers, BarChart2, Sparkles } from 'lucide-react';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';

export const About: React.FC = () => {
  return (
    <div className="max-w-4xl mx-auto space-y-12 py-4">
      {/* Page Title */}
      <div className="space-y-2">
        <Badge variant="emerald" className="text-xs font-mono">System Architecture</Badge>
        <h1 className="text-3xl font-extrabold text-white tracking-tight">About AgriVision AI</h1>
        <p className="text-gray-400 text-sm">
          A production-quality deep learning system built for leaf disease classification and explainable AI telemetry.
        </p>
      </div>

      {/* Dataset Summary */}
      <section className="space-y-4">
        <h2 className="text-xl font-bold text-white flex items-center gap-2">
          <Database className="w-5 h-5 text-emerald-400" /> PlantVillage Dataset
        </h2>
        <Card className="space-y-4">
          <p className="text-sm text-gray-400 leading-relaxed">
            The system is trained on the open-access <strong>PlantVillage dataset</strong> containing over 54,305 curated RGB leaf images across 38 class labels representing 14 distinct crop species (Apple, Tomato, Potato, Corn, Grape, etc.).
          </p>
          <div className="grid grid-cols-3 gap-4 border-t border-gray-800 pt-4 text-center">
            <div>
              <p className="text-xl font-mono font-bold text-emerald-400">70%</p>
              <p className="text-xs text-gray-500">Train Split</p>
            </div>
            <div>
              <p className="text-xl font-mono font-bold text-cyan-400">15%</p>
              <p className="text-xs text-gray-500">Validation Split</p>
            </div>
            <div>
              <p className="text-xl font-mono font-bold text-amber-400">15%</p>
              <p className="text-xs text-gray-500">Test Split</p>
            </div>
          </div>
        </Card>
      </section>

      {/* Model Architectures Comparison */}
      <section className="space-y-4">
        <h2 className="text-xl font-bold text-white flex items-center gap-2">
          <Cpu className="w-5 h-5 text-emerald-400" /> Model Architecture Benchmarks
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Card className="space-y-3">
            <Badge variant="emerald">Model 1: ResNet-18 Transfer</Badge>
            <h3 className="text-lg font-bold text-white">Transfer Learning (Recommended)</h3>
            <p className="text-xs text-gray-400 leading-relaxed">
              Pre-trained ImageNet backbone with frozen convolutional features and a custom fully-connected classifier head.
            </p>
            <ul className="text-xs text-gray-300 space-y-1 font-mono pt-2 border-t border-gray-800">
              <li>Test Accuracy: ~97.8%</li>
              <li>F1 Score: 0.97</li>
              <li>Parameters: ~11.2M</li>
            </ul>
          </Card>

          <Card className="space-y-3">
            <Badge variant="cyan">Model 2: Custom CNN Baseline</Badge>
            <h3 className="text-lg font-bold text-white">4-Stage Convolutional Network</h3>
            <p className="text-xs text-gray-400 leading-relaxed">
              Custom sequential network with 4 ConvBlocks (Conv + BatchNorm + ReLU + MaxPool), FC(512), Dropout(0.5), and Softmax.
            </p>
            <ul className="text-xs text-gray-300 space-y-1 font-mono pt-2 border-t border-gray-800">
              <li>Test Accuracy: ~88.4%</li>
              <li>F1 Score: 0.87</li>
              <li>Parameters: ~13.1M</li>
            </ul>
          </Card>
        </div>
      </section>

      {/* Grad-CAM Mechanics */}
      <section className="space-y-4">
        <h2 className="text-xl font-bold text-white flex items-center gap-2">
          <Sparkles className="w-5 h-5 text-emerald-400" /> Grad-CAM Explainability
        </h2>
        <Card className="space-y-3">
          <p className="text-sm text-gray-400 leading-relaxed">
            Gradient-weighted Class Activation Mapping (Grad-CAM) attaches backward hooks to the final convolutional layer of the ResNet backbone. By calculating the gradients of the score for class <em>c</em> with respect to feature activation maps, the model produces a coarse localization map highlighting leaf lesions.
          </p>
        </Card>
      </section>
    </div>
  );
};
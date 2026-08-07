import React from 'react';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { 
  Sparkles, 
  ArrowRight, 
  Activity, 
  Zap, 
  ShieldCheck, 
  Cpu, 
  Database, 
  Layers, 
  CheckCircle2, 
  Eye
} from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';

export const Home: React.FC = () => {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { staggerChildren: 0.15 }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.5 } }
  };

  return (
    <motion.div 
      className="space-y-24 py-6"
      variants={containerVariants}
      initial="hidden"
      animate="visible"
    >
      {/* Hero Section */}
      <section className="relative text-center max-w-4xl mx-auto space-y-8 pt-10">
        <div className="absolute -top-20 left-1/2 -translate-x-1/2 w-96 h-96 bg-emerald-500/10 rounded-full blur-3xl pointer-events-none -z-10" />
        
        <motion.div variants={itemVariants} className="inline-flex items-center">
          <Badge variant="emerald" className="gap-2 py-1.5 px-4 text-xs font-mono">
            <Sparkles className="w-3.5 h-3.5" /> PyTorch + ResNet-18 Transfer Learning
          </Badge>
        </motion.div>

        <motion.h1 
          variants={itemVariants}
          className="text-4xl sm:text-6xl font-extrabold tracking-tight text-white leading-tight"
        >
          Instant Crop Disease Diagnostics Powered by <span className="text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 via-teal-300 to-cyan-400">Deep Learning</span>
        </motion.h1>

        <motion.p 
          variants={itemVariants}
          className="text-lg text-gray-400 max-w-2xl mx-auto leading-relaxed"
        >
          Upload leaf photos to diagnose 38 plant disease classes across 14 crop species. 
          Get instant predictions, Grad-CAM heatmaps, and agricultural treatment plans.
        </motion.p>

        <motion.div variants={itemVariants} className="flex flex-col sm:flex-row gap-4 justify-center pt-2">
          <Link to="/predict">
            <Button size="lg" className="w-full sm:w-auto gap-2 text-base px-8 shadow-lg shadow-emerald-500/20">
              Run Leaf Diagnosis <ArrowRight className="w-4 h-4" />
            </Button>
          </Link>
          <Link to="/about">
            <Button size="lg" variant="outline" className="w-full sm:w-auto text-base px-8">
              Model Architecture & Metrics
            </Button>
          </Link>
        </motion.div>

        {/* Live Metrics Strip */}
        <motion.div 
          variants={itemVariants} 
          className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-10 border-t border-gray-800/80 text-left"
        >
          {[
            { label: 'Model Accuracy', value: '98.2%', sub: 'ResNet-18 Transfer' },
            { label: 'Inference Speed', value: '< 120ms', sub: 'GPU Accelerated' },
            { label: 'Dataset Images', value: '~54,305', sub: 'PlantVillage Corpus' },
            { label: 'Classes Supported', value: '38 Classes', sub: 'Across 14 Crops' }
          ].map((stat, i) => (
            <div key={i} className="p-4 rounded-xl bg-gray-900/40 border border-gray-800/50">
              <p className="text-2xl font-bold text-emerald-400 font-mono">{stat.value}</p>
              <p className="text-sm font-medium text-gray-200 mt-1">{stat.label}</p>
              <p className="text-xs text-gray-500">{stat.sub}</p>
            </div>
          ))}
        </motion.div>
      </section>

      {/* Workflow Pipeline Section */}
      <section className="space-y-12">
        <div className="text-center max-w-2xl mx-auto space-y-3">
          <h2 className="text-2xl sm:text-3xl font-bold text-white tracking-tight">How Prediction Works</h2>
          <p className="text-gray-400 text-sm">
            End-to-end Computer Vision inference pipeline designed for real-time agricultural telemetry.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 relative">
          {[
            {
              step: '01',
              title: 'Image Ingestion',
              desc: 'Drag & drop leaf photo. Client validates file resolution and formats to RGB.',
              icon: Cpu
            },
            {
              step: '02',
              title: 'CNN Inference',
              desc: 'Image is resized to 224x224 and processed via ResNet-18 or Custom CNN Backbone.',
              icon: Layers
            },
            {
              step: '03',
              title: 'Grad-CAM Explainability',
              desc: 'PyTorch gradient hooks compute spatial heatmaps highlighting infected regions.',
              icon: Eye
            },
            {
              step: '04',
              title: 'Actionable Insights',
              desc: 'Returns top predictions, confidence score, and targeted chemical/organic treatments.',
              icon: ShieldCheck
            }
          ].map((item, idx) => (
            <Card key={idx} className="relative overflow-hidden group hover:border-emerald-500/40 transition-all">
              <span className="absolute top-4 right-4 text-4xl font-black text-gray-800/50 group-hover:text-emerald-500/20 transition-colors font-mono">
                {item.step}
              </span>
              <div className="p-3 rounded-lg bg-emerald-500/10 text-emerald-400 w-fit mb-4">
                <item.icon className="w-5 h-5" />
              </div>
              <h3 className="text-lg font-semibold text-white mb-2">{item.title}</h3>
              <p className="text-sm text-gray-400 leading-relaxed">{item.desc}</p>
            </Card>
          ))}
        </div>
      </section>

      {/* Model Capabilities Grid */}
      <section className="space-y-8">
        <div className="text-center max-w-2xl mx-auto space-y-3">
          <h2 className="text-2xl sm:text-3xl font-bold text-white tracking-tight">System Capabilities</h2>
          <p className="text-gray-400 text-sm">Engineered for accuracy, explainability, and actionable recommendations.</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <Card className="space-y-4">
            <div className="p-3 rounded-lg bg-cyan-500/10 text-cyan-400 w-fit">
              <Activity className="w-6 h-6" />
            </div>
            <h3 className="text-xl font-bold text-white">Top-5 Probability Distribution</h3>
            <p className="text-gray-400 text-sm leading-relaxed">
              Provides granular confidence scores across all candidate classes, enabling agronomists to evaluate potential co-infections or borderline classifications.
            </p>
          </Card>

          <Card className="space-y-4">
            <div className="p-3 rounded-lg bg-emerald-500/10 text-emerald-400 w-fit">
              <Eye className="w-6 h-6" />
            </div>
            <h3 className="text-xl font-bold text-white">Grad-CAM Spatial Explainability</h3>
            <p className="text-gray-400 text-sm leading-relaxed">
              Uses Gradient-weighted Class Activation Mapping to project visual heatmaps directly onto the leaf, validating where the neural network is focusing.
            </p>
          </Card>

          <Card className="space-y-4">
            <div className="p-3 rounded-lg bg-amber-500/10 text-amber-400 w-fit">
              <Zap className="w-6 h-6" />
            </div>
            <h3 className="text-xl font-bold text-white">Agronomic Treatment Protocols</h3>
            <p className="text-gray-400 text-sm leading-relaxed">
              Maps predicted disease categories to tailored therapeutic guidelines, providing organic, chemical, and preventive measures.
            </p>
          </Card>
        </div>
      </section>

      {/* Call to Action Banner */}
      <Card className="p-10 text-center bg-gradient-to-r from-emerald-950/40 via-gray-900 to-cyan-950/40 border-emerald-500/30 space-y-6">
        <h2 className="text-3xl font-bold text-white tracking-tight">Ready to test your crop images?</h2>
        <p className="text-gray-400 max-w-xl mx-auto text-sm">
          Run inference in real-time or enable prototype mode to test with instant sample data.
        </p>
        <Link to="/predict" className="inline-block">
          <Button size="lg" className="px-8 gap-2">
            Start Leaf Diagnosis Now <ArrowRight className="w-4 h-4" />
          </Button>
        </Link>
      </Card>
    </motion.div>
  );
};
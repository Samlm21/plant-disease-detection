import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Search, Trash2, Download, Calendar, Clock, Filter, AlertCircle, FileText } from 'lucide-react';
import { Card } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { historyService } from '@/services/historyService';
import { HistoryItem } from '@/types/prediction';

export const History: React.FC = () => {
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCrop, setSelectedCrop] = useState<string>('All');

  useEffect(() => {
    setHistory(historyService.getAll());
  }, []);

  const handleDelete = (id: string) => {
    historyService.delete(id);
    setHistory(historyService.getAll());
  };

  const handleClearAll = () => {
    if (confirm('Are you sure you want to clear all prediction history?')) {
      historyService.clear();
      setHistory([]);
    }
  };

  const filteredHistory = history.filter((item) => {
    const matchesSearch = item.diseaseInfo.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                          item.diseaseInfo.crop.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesCrop = selectedCrop === 'All' || item.diseaseInfo.crop === selectedCrop;
    return matchesSearch && matchesCrop;
  });

  const crops = ['All', ...Array.from(new Set(history.map((i) => i.diseaseInfo.crop)))];

  return (
    <div className="max-w-6xl mx-auto space-y-8 py-4">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-extrabold text-white tracking-tight">Diagnosis History</h1>
          <p className="text-gray-400 text-sm">Review previously analyzed leaf diagnostics stored locally.</p>
        </div>
        {history.length > 0 && (
          <Button variant="danger" size="sm" onClick={handleClearAll} className="gap-2 self-start sm:self-auto">
            <Trash2 className="w-4 h-4" /> Clear All History
          </Button>
        )}
      </div>

      {/* Filter and Search Bar */}
      <Card className="p-4 flex flex-col md:flex-row items-center gap-4">
        <div className="relative flex-1 w-full">
          <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
          <input
            type="text"
            placeholder="Search by disease or crop..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full bg-gray-950 border border-gray-800 rounded-lg pl-9 pr-4 py-2 text-sm text-gray-200 focus:outline-none focus:border-emerald-500/50"
          />
        </div>

        <div className="flex items-center gap-2 w-full md:w-auto">
          <Filter className="w-4 h-4 text-gray-500" />
          <select
            value={selectedCrop}
            onChange={(e) => setSelectedCrop(e.target.value)}
            className="bg-gray-950 border border-gray-800 rounded-lg px-3 py-2 text-sm text-gray-300 focus:outline-none focus:border-emerald-500/50"
          >
            {crops.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
        </div>
      </Card>

      {/* History Cards Grid */}
      {filteredHistory.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredHistory.map((item) => (
            <Card key={item.id} className="space-y-4 flex flex-col justify-between">
              <div className="space-y-3">
                <div className="flex items-center justify-between text-xs text-gray-500 font-mono">
                  <span className="flex items-center gap-1">
                    <Calendar className="w-3.5 h-3.5" />
                    {new Date(item.createdAt).toLocaleDateString()}
                  </span>
                  <span className="flex items-center gap-1">
                    <Clock className="w-3.5 h-3.5" />
                    {item.inferenceTimeMs}ms
                  </span>
                </div>

                <div className="flex gap-3">
                  {item.imageUrl && (
                    <img 
                      src={item.imageUrl} 
                      alt={item.diseaseInfo.name} 
                      className="w-16 h-16 rounded-lg object-cover border border-gray-800 flex-shrink-0"
                    />
                  )}
                  <div className="space-y-1">
                    <Badge variant="emerald" className="text-[10px]">{item.diseaseInfo.crop}</Badge>
                    <h3 className="text-sm font-bold text-white line-clamp-1">{item.diseaseInfo.name}</h3>
                    <p className="text-xs text-emerald-400 font-mono">
                      Confidence: {(item.confidence * 100).toFixed(1)}%
                    </p>
                  </div>
                </div>
              </div>

              <div className="pt-2 border-t border-gray-800 flex items-center justify-between">
                <span className="text-xs text-gray-500">ID: {item.id.slice(0, 8)}</span>
                <button
                  onClick={() => handleDelete(item.id)}
                  className="p-1.5 text-gray-500 hover:text-rose-400 transition-colors"
                  title="Delete record"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            </Card>
          ))}
        </div>
      ) : (
        <Card className="p-12 text-center space-y-3 border-dashed border-gray-800">
          <FileText className="w-10 h-10 text-gray-600 mx-auto" />
          <h3 className="text-base font-semibold text-gray-300">No Prediction Records Found</h3>
          <p className="text-xs text-gray-500">Run leaf diagnoses to build your local prediction history.</p>
        </Card>
      )}
    </div>
  );
};
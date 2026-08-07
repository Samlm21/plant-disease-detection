import React from 'react';
import { Link } from 'react-router-dom';
import { AlertCircle, ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/Button';

export const NotFound: React.FC = () => {
  return (
    <div className="min-h-[60vh] flex flex-col items-center justify-center space-y-6 text-center">
      <div className="p-4 rounded-full bg-rose-500/10 text-rose-400">
        <AlertCircle className="w-12 h-12" />
      </div>
      <div className="space-y-2">
        <h1 className="text-4xl font-extrabold text-white tracking-tight">404</h1>
        <p className="text-gray-400">The page you are looking for does not exist or has been moved.</p>
      </div>
      <Link to="/">
        <Button variant="outline" className="gap-2">
          <ArrowLeft className="w-4 h-4" /> Return to Dashboard
        </Button>
      </Link>
    </div>
  );
};

export default NotFound;
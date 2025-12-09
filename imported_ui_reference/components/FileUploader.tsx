
import React, { useCallback, useState } from 'react';

interface FileUploaderProps {
  onUpload: (files: File[]) => void;
  isLoading: boolean;
  label: string;
  generatingLabel: string;
  buttonLabel: string;
}

export const FileUploader: React.FC<FileUploaderProps> = ({ 
    onUpload, 
    isLoading, 
    label,
    generatingLabel,
    buttonLabel
}) => {
  const [files, setFiles] = useState<File[]>([]);
  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      setFiles(Array.from(e.dataTransfer.files));
    }
  }, []);

  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFiles(Array.from(e.target.files));
    }
  }, []);

  const handleGenerate = () => {
      if (files.length > 0) {
          onUpload(files);
          setFiles([]); // Clear after upload starts
      }
  };

  return (
    <div className="flex flex-col gap-2">
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`
          relative border-2 border-dashed rounded-md p-4 transition-colors cursor-pointer text-center
          ${isDragging ? 'border-cyan-400 bg-cyan-900/20' : 'border-gray-600 bg-gray-900/40 hover:border-gray-500'}
          ${isLoading ? 'opacity-50 pointer-events-none' : ''}
        `}
      >
        <input
          type="file"
          multiple
          onChange={handleFileChange}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          disabled={isLoading}
          accept=".txt,.md,.pdf,.csv,.json"
        />
        
        {files.length > 0 ? (
            <div className="flex flex-col gap-1">
                {files.map((f, idx) => (
                    <span key={idx} className="text-xs text-cyan-300 truncate font-mono">
                        {f.name} ({(f.size / 1024).toFixed(1)} KB)
                    </span>
                ))}
            </div>
        ) : (
            <div className="text-gray-400 text-xs">
                <p>{label}</p>
                <p className="text-[10px] text-gray-500 mt-1">PDF, TXT, MD, CSV</p>
            </div>
        )}
      </div>
      
      <button
        onClick={handleGenerate}
        disabled={files.length === 0 || isLoading}
        className="w-full py-1.5 bg-cyan-800 hover:bg-cyan-700 text-cyan-100 text-xs font-bold rounded disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
      >
        {isLoading ? generatingLabel : buttonLabel}
      </button>
    </div>
  );
};

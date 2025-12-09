import React, { useRef, useState } from 'react';
import { UploadedFile } from '../types';

interface FileUploaderProps {
    onFileSelect: (file: UploadedFile | null) => void;
    selectedFile: UploadedFile | null;
}

const FileUploader: React.FC<FileUploaderProps> = ({ onFileSelect, selectedFile }) => {
    const fileInputRef = useRef<HTMLInputElement>(null);
    const [isDragging, setIsDragging] = useState(false);

    const processFile = (file: File) => {
        if (!file) return;
        
        // Support PDF, Images, and Excel
        const validTypes = [
            'application/pdf', 
            'image/png', 
            'image/jpeg', 
            'image/webp',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'application/vnd.ms-excel'
        ];
        
        // Check extension as fallback for Excel
        const isExcel = file.name.endsWith('.xlsx') || file.name.endsWith('.xls') || file.type.includes('sheet') || file.type.includes('excel');

        if (!validTypes.includes(file.type) && !isExcel) {
            alert('Formato non supportato. Carica PDF, Excel (XLSX), PNG o JPEG.');
            return;
        }

        if (isExcel) {
            alert("Excel parsing requires server-side processing or full library support. Please use PDF or Images for this demo.");
            return;
        }

        // Gestione Standard per Immagini e PDF
        const reader = new FileReader();
        reader.onload = (e) => {
            const base64 = e.target?.result as string;
            onFileSelect({
                name: file.name,
                type: file.type,
                data: base64
            });
        };
        reader.readAsDataURL(file);
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            processFile(e.dataTransfer.files[0]);
        }
    };

    return (
        <div className="mb-4">
            <input
                type="file"
                ref={fileInputRef}
                onChange={(e) => e.target.files && processFile(e.target.files[0])}
                className="hidden"
                accept=".pdf,.png,.jpg,.jpeg,.webp,.xlsx,.xls"
            />
            
            {!selectedFile ? (
                <div
                    onClick={() => fileInputRef.current?.click()}
                    onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                    onDragLeave={() => setIsDragging(false)}
                    onDrop={handleDrop}
                    className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-all
                        ${isDragging 
                            ? 'border-cyan-400 bg-cyan-900/20' 
                            : 'border-slate-600 hover:border-cyan-500 hover:bg-slate-700/50'}`}
                >
                    <div className="flex flex-col items-center gap-2 text-slate-400">
                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-8 h-8 mb-1">
                            <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
                        </svg>
                        <span className="text-sm font-medium">Clicca o trascina per analizzare documenti</span>
                        <span className="text-xs text-slate-500">(PDF, Images, Excel*)</span>
                    </div>
                </div>
            ) : (
                <div className="bg-cyan-900/30 border border-cyan-700/50 rounded-lg p-3 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <div className="bg-cyan-900/50 p-2 rounded text-cyan-300">
                            {selectedFile.name.endsWith('.xlsx') || selectedFile.name.endsWith('.csv') ? (
                                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5">
                                  <path strokeLinecap="round" strokeLinejoin="round" d="M3.375 19.5h17.25m-17.25 0a1.125 1.125 0 01-1.125-1.125M3.375 19.5h7.5c.621 0 1.125-.504 1.125-1.125m-9.75 0V5.625m0 12.75v-1.5c0-.621.504-1.125 1.125-1.125m13.5 2.625h3a1.125 1.125 0 001.125-1.125v-1.5c0-.621-.504-1.125-1.125-1.125H11.25" />
                                </svg>
                            ) : (
                                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5">
                                    <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
                                </svg>
                            )}
                        </div>
                        <div className="text-sm text-slate-200">
                            <p className="font-medium truncate max-w-[200px]">{selectedFile.name}</p>
                            <p className="text-xs text-slate-400 uppercase">{selectedFile.type.split('/')[1] || 'DATA'}</p>
                        </div>
                    </div>
                    <button 
                        onClick={(e) => { e.stopPropagation(); onFileSelect(null); }}
                        className="text-slate-400 hover:text-red-400 transition-colors p-1"
                    >
                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5">
                            <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                        </svg>
                    </button>
                </div>
            )}
        </div>
    );
};

export default FileUploader;
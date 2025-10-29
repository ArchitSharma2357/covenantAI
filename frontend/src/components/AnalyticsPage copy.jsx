import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import UploadBox from './UploadBox';
import AnalyzeButton from './AnalyzeButton';
import {
	Download,
	TrendingUp,
	AlertTriangle,
	CheckCircle,
	Lightbulb,
	Loader2,
	ChevronDown,
	FileText,
} from 'lucide-react';

/**
 * AnalyticsPage
 * - Premium, frosted-glass styled analytics page.
 * - Supports file/text input (UploadBox), analysis, and export.
 * - Left nav with sections, right content area is scrollable (min-h-0 on parents)
 */
const AnalyticsPage = () => {
	const navigate = useNavigate();
	const [selectedFile, setSelectedFile] = useState(null);
	const [textInput, setTextInput] = useState('');
	const [analysisResult, setAnalysisResult] = useState(null);
	const [isAnalyzing, setIsAnalyzing] = useState(false);
	const [isDownloading, setIsDownloading] = useState(false);
	const [uploadProgress, setUploadProgress] = useState(0);
	const [activeSection, setActiveSection] = useState('summary');
	const [showDownloadMenu, setShowDownloadMenu] = useState(false);

	const sections = [
		{ key: 'summary', label: 'Summary', icon: FileText },
		{ key: 'risk', label: 'Risk Assessment', icon: TrendingUp },
		{ key: 'swot', label: 'SWOT', icon: Lightbulb },
		{ key: 'flags', label: 'Critical Flags', icon: AlertTriangle },
		{ key: 'negotiation', label: 'Negotiation Plan', icon: CheckCircle },
		{ key: 'benchmarks', label: 'Benchmarks', icon: TrendingUp },
	];

	const handleFileSelect = (file) => setSelectedFile(file);
	const handleTextInput = (text) => setTextInput(text);

	const handleGenerateReport = async () => {
		if (!selectedFile && !textInput) {
			window.dispatchEvent(new CustomEvent('notifyEvent', { detail: 'Select a file or enter text to analyze.' }));
			return;
		}

		setIsAnalyzing(true);
		setUploadProgress(0);

		const formData = new FormData();
		if (selectedFile) {
			// if browser File object, append directly
			formData.append('file', selectedFile);
		} else {
			formData.append('rawText', textInput);
		}

		try {
			// POST to backend analyze endpoint
			const resp = await axios.post('http://localhost:8000/api/analyze', formData, {
				headers: { 'Content-Type': 'multipart/form-data' },
				onUploadProgress: (e) => {
					if (!e.lengthComputable) return;
					const pct = Math.round((e.loaded * 100) / e.total);
					setUploadProgress(pct);
				},
			});

			setAnalysisResult(resp.data);
			setUploadProgress(100);

			// Save guest history locally if unauthenticated
			const token = localStorage.getItem('token');
			if (!token) {
				try {
					const guest = JSON.parse(localStorage.getItem('guest_history_v1') || '[]');
					const id = 'g-' + Date.now();
					guest.unshift({ id, filename: selectedFile?.name || 'Pasted Text', upload_time: new Date().toISOString(), analysis: resp.data });
					localStorage.setItem('guest_history_v1', JSON.stringify(guest.slice(0, 20)));
				} catch (e) {
					// ignore
				}
			}

			// navigate to report if desired (kept for compatibility)
			// navigate('/report', { state: { analysisResult: resp.data } });
		} catch (err) {
			console.error('Analyze error', err);
			window.dispatchEvent(new CustomEvent('notifyEvent', { detail: 'Failed to analyze document.' }));
		} finally {
			setIsAnalyzing(false);
		}
	};

	const handleDownloadPDF = async () => {
		if (!analysisResult) return;
		setIsDownloading(true);
		try {
			const resp = await axios.post('http://localhost:8000/api/export/pdf', analysisResult, { responseType: 'blob' });
			if (resp.status === 200) {
				const url = window.URL.createObjectURL(new Blob([resp.data]));
				const a = document.createElement('a');
				a.href = url;
				a.download = 'legal-analysis.pdf';
				document.body.appendChild(a);
				a.click();
				a.remove();
				window.dispatchEvent(new CustomEvent('notifyEvent', { detail: 'Downloaded PDF report' }));
			}
		} catch (e) {
			console.error('Download error', e);
			window.dispatchEvent(new CustomEvent('notifyEvent', { detail: 'Failed to download PDF' }));
		} finally {
			setIsDownloading(false);
		}
	};

	useEffect(() => {
		const onAnalyzeFile = async (e) => {
			const file = e.detail;
			setSelectedFile(file);
			setIsAnalyzing(true);
			try {
				const formData = new FormData();
				formData.append('file', file);
				const resp = await axios.post('http://localhost:8000/api/analyze', formData);
				setAnalysisResult(resp.data);
			} catch (err) {
				console.warn('analyzeFile handler failed', err);
			} finally {
				setIsAnalyzing(false);
			}
		};
		window.addEventListener('analyzeFile', onAnalyzeFile);
		return () => window.removeEventListener('analyzeFile', onAnalyzeFile);
	}, []);

	// close download menu on outside click
	useEffect(() => {
		const onDocClick = (e) => {
			const el = e.target;
			if (!el) return;
			if (!el.closest) return;
			if (!el.closest('.download-menu-anchor')) {
				setShowDownloadMenu(false);
			}
		};
		if (showDownloadMenu) document.addEventListener('click', onDocClick);
		return () => document.removeEventListener('click', onDocClick);
	}, [showDownloadMenu]);

	const renderSection = () => {
		if (!analysisResult) return null;
		switch (activeSection) {
			case 'summary':
				return (
					<div className="p-6">
						<h3 className="text-xl font-semibold mb-3">Executive Summary</h3>
						<p className="text-sm text-muted-foreground whitespace-pre-wrap">{analysisResult.summary || 'No summary available.'}</p>
					</div>
				);
			case 'risk':
				return (
					<div className="p-6">
						<h3 className="text-xl font-semibold mb-3">Risk Assessment</h3>
						<div className="text-sm text-muted-foreground">{analysisResult.riskScore?.description || 'No risk data.'}</div>
					</div>
				);
			case 'swot':
				return (
					<div className="p-6">
						<h3 className="text-xl font-semibold mb-4">SWOT</h3>
						<div className="grid md:grid-cols-2 gap-4">
							<div className="bg-gray-50 rounded-lg p-4">
								<h4 className="font-medium">Strengths</h4>
								<ol className="list-decimal ml-5 mt-2 text-sm space-y-1">{(analysisResult.analysis?.strengths || []).map((s, i) => <li key={i}>{s.text}</li>)}</ol>
							</div>
							<div className="bg-gray-50 rounded-lg p-4">
								<h4 className="font-medium">Weaknesses</h4>
								<ol className="list-decimal ml-5 mt-2 text-sm space-y-1">{(analysisResult.analysis?.weaknesses || []).map((s, i) => <li key={i}>{s.text}</li>)}</ol>
							</div>
						</div>
					</div>
				);
			case 'flags':
				return (
					<div className="p-6">
						<h3 className="text-xl font-semibold mb-3">Critical Flags</h3>
						<div className="space-y-3">
							{(analysisResult.criticalFlags || []).map((f, i) => (
								<div key={i} className="p-3 bg-red-50 rounded-lg border border-red-100">
									<div className="font-semibold text-red-600">{f.title}</div>
									<div className="text-sm mt-1">{f.explanation}</div>
								</div>
							))}
						</div>
					</div>
				);
			case 'negotiation':
				return (
					<div className="p-6">
						<h3 className="text-xl font-semibold mb-3">Negotiation Plan</h3>
						<ol className="list-decimal ml-5 text-sm space-y-2">{(analysisResult.negotiationPoints || []).map((p, i) => <li key={i}>{p.title}: {p.example}</li>)}</ol>
					</div>
				);
			case 'benchmarks':
				return (
					<div className="p-6">
						<h3 className="text-xl font-semibold mb-3">Benchmarks</h3>
						<div className="grid md:grid-cols-3 gap-4">
							<div className="bg-gray-50 rounded-lg p-4 text-center">
								<div className="text-2xl font-bold">{Math.round(((analysisResult.riskScore?.score || 0) / (analysisResult.riskScore?.max || 1)) * 100)}%</div>
								<div className="text-sm text-muted-foreground">Risk vs Avg</div>
							</div>
							<div className="bg-gray-50 rounded-lg p-4 text-center">
								<div className="text-2xl font-bold">{(analysisResult.analysis?.strengths || []).length}</div>
								<div className="text-sm text-muted-foreground">Strengths</div>
							</div>
							<div className="bg-gray-50 rounded-lg p-4 text-center">
								<div className="text-2xl font-bold">{(analysisResult.criticalFlags || []).length}</div>
								<div className="text-sm text-muted-foreground">Critical Issues</div>
							</div>
						</div>
					</div>
				);
			default:
				return null;
		}
	};

		// initial empty state (no analysis yet)
		if (!analysisResult) {
			return (
				<div className="min-h-screen bg-black p-6">
					<div className="max-w-4xl mx-auto">
						<div className="bg-white/4 backdrop-blur-md rounded-2xl border border-white/6 shadow-lg overflow-hidden">
							<div className="p-8">
								<div className="flex items-start justify-between">
									<div>
										<h1 className="text-3xl font-extrabold text-white">CovenantAI — Analysis</h1>
										<p className="text-sm text-gray-300 mt-2">Upload a legal document or paste text to get clear, AI-powered insights.</p>
									</div>
									<div className="flex items-center gap-3">
										{/* header Upload removed per request */}
									</div>
								</div>

								<div className="mt-8 flex flex-col items-center justify-center min-h-[40vh]">
									<div className="w-full max-w-2xl">
										<UploadBox onFileSelect={handleFileSelect} onTextInput={handleTextInput} className="bg-white/6" />
										<div className="mt-6 flex justify-center">
											<AnalyzeButton onAnalyze={handleGenerateReport} loading={isAnalyzing} className="w-full md:w-1/2 bg-white text-black" />
										</div>
										{uploadProgress > 0 && uploadProgress < 100 && (
											<div className="mt-4">
												<div className="w-full bg-white/10 rounded-full h-2 overflow-hidden">
													<div className="h-2 bg-white" style={{ width: `${uploadProgress}%` }} />
												</div>
											</div>
										)}
									</div>
								</div>
							</div>
						</div>
					</div>
				</div>
			);
		}

	return (
		<div className="min-h-screen bg-gradient-to-b from-gray-50 to-white p-8">
			<div className="max-w-6xl mx-auto">
				<div className="relative bg-white/60 backdrop-blur-md rounded-2xl border border-white/30 shadow-2xl overflow-hidden min-h-[72vh] flex flex-col">
								<div className="p-8 border-b border-white/12">
						<div className="flex items-start justify-between">
							<div>
											<h1 className="text-3xl font-extrabold text-white">CovenantAI — Analysis</h1>
											<p className="text-sm text-gray-300 mt-2 max-w-2xl">A clear, executive-ready summary and detailed breakdown of risks, negotiation points, and flags.</p>
							</div>
							<div className="flex items-center gap-3">
											<button onClick={handleDownloadPDF} disabled={isDownloading} className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-white/8 border border-white/12">
												{isDownloading ? <Loader2 className="animate-spin text-white" size={16} /> : <Download size={16} className="text-white" />}
												<span className="hidden sm:inline text-white">Export</span>
											</button>
								<div className="relative download-menu-anchor">
									<button onClick={() => setShowDownloadMenu(s => !s)} className="p-2 rounded-lg bg-white/70 border border-white/20">
										<ChevronDown size={14} />
									</button>
									{showDownloadMenu && (
										<div className="absolute right-0 mt-2 w-44 bg-white/95 rounded-lg border border-white/20 shadow z-40">
											<button onClick={async () => { setShowDownloadMenu(false); await handleDownloadPDF(); }} className="w-full text-left px-3 py-2 hover:bg-gray-50">Download PDF</button>
											<button onClick={async () => { setShowDownloadMenu(false); /* bundle */ await handleDownloadPDF(); }} className="w-full text-left px-3 py-2 hover:bg-gray-50">Download Bundle</button>
										</div>
									)}
								</div>
							</div>
						</div>
					</div>

					<div className="flex-1 min-h-0 flex">
						{/* Left nav */}
						<aside className="w-64 border-r border-white/20 p-6 overflow-auto">
							<div className="space-y-4">
								{sections.map((s) => {
									const Icon = s.icon;
									const active = activeSection === s.key;
									return (
										<button key={s.key} onClick={() => setActiveSection(s.key)} className={`w-full text-left flex items-center gap-3 p-3 rounded-lg ${active ? 'bg-white/30' : 'hover:bg-white/10'}`}>
											<Icon size={18} />
											<div>
												<div className="text-sm font-medium">{s.label}</div>
											</div>
										</button>
									);
								})}
							</div>
						</aside>

						{/* Right content: make scrollable and fill space */}
						<main className="flex-1 min-h-0 overflow-auto p-6">
							<div className="max-w-none bg-white rounded-xl shadow-sm border border-white/10 h-full">
								{renderSection()}
							</div>
						</main>
					</div>
				</div>
			</div>
		</div>
	);
};

export default AnalyticsPage;

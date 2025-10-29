import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import UploadBox from './UploadBox';
import AnalyzeButton from './AnalyzeButton';
import { Download, TrendingUp, AlertTriangle, CheckCircle, Lightbulb, Loader2, ChevronDown, FileText } from 'lucide-react';

/**
 * AnalyticsPage
 * - Uses the app's theme tokens and component patterns (bg-background, bg-card, text-foreground, etc.)
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
		if (selectedFile) formData.append('file', selectedFile);
		else formData.append('rawText', textInput);

		try {
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

				const token = localStorage.getItem('token');
				if (!token) {
					try {
						// Use a consistent lightweight shape that HistoryPage expects
						const guest = JSON.parse(localStorage.getItem('guest_history_v1') || '[]');
						const id = 'g-' + Date.now();
						const entry = {
							id,
							filename: selectedFile?.name || 'Pasted Text',
							upload_time: new Date().toISOString(),
							expires_at: new Date(Date.now() + 10 * 60 * 1000).toISOString(),
							analysis_result: resp.data,
						};
						guest.unshift(entry);
						localStorage.setItem('guest_history_v1', JSON.stringify(guest.slice(0, 20)));
					} catch (e) {
						console.warn('Failed to write guest history to localStorage', e);
					}
				}
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
				const fd = new FormData();
				fd.append('file', file);
				const resp = await axios.post('http://localhost:8000/api/analyze', fd);
				setAnalysisResult(resp.data);
			} catch (err) {
				console.warn('analyzeFile failed', err);
			} finally {
				setIsAnalyzing(false);
			}
		};
		window.addEventListener('analyzeFile', onAnalyzeFile);
		return () => window.removeEventListener('analyzeFile', onAnalyzeFile);
	}, []);

	useEffect(() => {
		const onDocClick = (e) => {
			const el = e.target;
			if (!el || !el.closest) return;
			if (!el.closest('.download-menu-anchor')) setShowDownloadMenu(false);
		};
		if (showDownloadMenu) document.addEventListener('click', onDocClick);
		return () => document.removeEventListener('click', onDocClick);
	}, [showDownloadMenu]);

	const renderSection = () => {
		if (!analysisResult) return null;
			// normalize risk object from backend variations
			const risk = analysisResult.riskScore || analysisResult.risk_score || analysisResult.riskAssessment || analysisResult.risk_assessment || {};
		switch (activeSection) {
			case 'summary':
				return (
					<div className="p-6">
						<h3 className="text-xl font-semibold mb-3">Executive Summary</h3>
						<p className="text-sm text-muted-foreground whitespace-pre-wrap mb-4">{analysisResult.summary || 'No summary available.'}</p>
						{/* Document stats removed per user request */}
					</div>
				);
			case 'risk':
				return (
					<div className="p-6">
						<div className="flex items-center justify-between mb-4">
							<h3 className="text-xl font-semibold">Risk Assessment</h3>
							<div className="flex items-center gap-3">
								{/* Risk counter */}
								{(() => {
									const score = Number(risk.score || risk.value || 0) || 0;
									const max = Number(risk.max || risk.scale || 100) || 100;
									const pct = Math.round((score / (max || 1)) * 100);
									let tone = 'bg-emerald-500 text-emerald-50';
									let label = 'Low';
									if (pct > 66) { tone = 'bg-destructive text-destructive-foreground'; label = 'High'; }
									else if (pct > 33) { tone = 'bg-amber-500 text-amber-900'; label = 'Medium'; }
									return (
										<div className={`flex items-center gap-3 px-3 py-1 rounded-full ${tone}`}> 
											<div className="text-sm font-semibold">{pct}%</div>
											<div className="text-xs opacity-90">{label}</div>
										</div>
									);
								})()}
								{/* Acknowledge / Back button */}
								<button onClick={() => navigate(-1)} className="px-3 py-1 rounded-md bg-background border border-border text-sm">Acknowledge & Back</button>
							</div>
							</div>
							<div className="text-sm text-muted-foreground mb-4">{risk.description || risk.label || risk.summary || 'No risk data.'}</div>
							{/* Progress bar / details */}
							<div className="mb-4">
								<div className="w-full bg-muted/30 rounded-full h-3 overflow-hidden">
									<div className="h-3" style={{ width: `${Math.round(((risk.score || 0) / (risk.max || 1)) * 100)}%`, background: (function(){ const pct = Math.round(((risk.score || 0)/(risk.max || 1))*100); if(pct>66) return 'rgb(220,38,38)'; if(pct>33) return 'rgb(245,158,11)'; return 'rgb(34,197,94)'; })() }} />
								</div>
							</div>
							{/* More granular details if available */}
							<div className="grid md:grid-cols-3 gap-4">
								<div className="bg-muted/50 rounded-lg p-3">
									<div className="text-xs text-muted-foreground">Score</div>
									<div className="font-semibold">{risk.score ?? risk.value ?? '—'}</div>
								</div>
								<div className="bg-muted/50 rounded-lg p-3">
									<div className="text-xs text-muted-foreground">Scale / Max</div>
									<div className="font-semibold">{risk.max ?? risk.scale ?? '—'}</div>
								</div>
								<div className="bg-muted/50 rounded-lg p-3">
									<div className="text-xs text-muted-foreground">Level</div>
									<div className="font-semibold">{risk.level || risk.severity || (Math.round(((risk.score || 0)/(risk.max || 1))*100) > 66 ? 'High' : (Math.round(((risk.score || 0)/(risk.max || 1))*100) > 33 ? 'Medium' : 'Low'))}</div>
								</div>
							</div>
						</div>
					);
			case 'swot':
				return (
					<div className="p-6">
						<h3 className="text-xl font-semibold mb-4">SWOT</h3>
						<div className="grid md:grid-cols-2 gap-4">
							<div className="bg-muted/50 rounded-lg p-4">
								<h4 className="font-medium">Strengths</h4>
								<ol className="list-decimal ml-5 mt-2 text-sm space-y-1">{(analysisResult.analysis?.strengths || []).map((s, i) => <li key={i}>{s.text}</li>)}</ol>
							</div>
							<div className="bg-muted/50 rounded-lg p-4">
								<h4 className="font-medium">Weaknesses</h4>
								<ol className="list-decimal ml-5 mt-2 text-sm space-y-1">{(analysisResult.analysis?.weaknesses || []).map((s, i) => <li key={i}>{s.text}</li>)}</ol>
							</div>
						</div>
						<div className="grid md:grid-cols-2 gap-4 mt-4">
							<div className="bg-muted/50 rounded-lg p-4">
								<h4 className="font-medium">Opportunities</h4>
								<ol className="list-decimal ml-5 mt-2 text-sm space-y-1">{(analysisResult.analysis?.opportunities || []).map((s, i) => <li key={i}>{s.text}</li>)}</ol>
							</div>
							<div className="bg-muted/50 rounded-lg p-4">
								<h4 className="font-medium">Threats</h4>
								<ol className="list-decimal ml-5 mt-2 text-sm space-y-1">{(analysisResult.analysis?.threats || []).map((s, i) => <li key={i}>{s.text}</li>)}</ol>
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
								<div key={i} className="p-3 bg-destructive/10 rounded-lg border border-destructive/20">
									<div className="font-semibold text-destructive">{f.title}</div>
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
							<div className="bg-muted/50 rounded-lg p-4 text-center">
								<div className="text-2xl font-bold">{Math.round(((risk.score || 0) / (risk.max || 1)) * 100)}%</div>
								<div className="text-sm text-muted-foreground">Risk vs Avg</div>
							</div>
							<div className="bg-muted/50 rounded-lg p-4 text-center">
								<div className="text-2xl font-bold">{(analysisResult.analysis?.strengths || []).length}</div>
								<div className="text-sm text-muted-foreground">Strengths</div>
							</div>
							<div className="bg-muted/50 rounded-lg p-4 text-center">
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
			<div className="min-h-screen bg-background p-6">
				<div className="max-w-4xl mx-auto">
					  <div className="bg-card rounded-lg border border-border shadow-sm min-h-[60vh] flex flex-col">
						<div className="p-6 border-b border-border">
							<div>
								<h1 className="text-2xl font-semibold text-foreground">CovenantAI — Analysis</h1>
								<p className="text-muted-foreground text-sm mt-1">Upload a legal document or paste text to get clear, AI-powered insights.</p>
							</div>
						</div>

						<div className="flex-1 p-6 overflow-auto flex items-center justify-center">
							<div className="w-full max-w-2xl">
								<div className="bg-background p-4 border border-border rounded-lg">
									<UploadBox onFileSelect={handleFileSelect} onTextInput={handleTextInput} />
								</div>

								<div className="mt-6 flex justify-center">
									<AnalyzeButton onAnalyze={handleGenerateReport} loading={isAnalyzing} className="w-full md:w-1/2 bg-primary text-primary-foreground" />
								</div>

								{uploadProgress > 0 && uploadProgress < 100 && (
									<div className="mt-4">
										<div className="w-full bg-muted/50 rounded-full h-2 overflow-hidden">
											<div className="h-2 bg-primary" style={{ width: `${uploadProgress}%` }} />
										</div>
									</div>
								)}
							</div>
						</div>

						{/* footer upload button removed — UploadBox contains upload controls */}
					</div>
				</div>
			</div>
		);
	}

	return (
		<div className="min-h-screen bg-background p-8">
			<div className="max-w-6xl mx-auto">
				<div className="relative bg-card rounded-2xl border border-border shadow-2xl overflow-hidden min-h-[72vh] flex flex-col">
					<div className="p-6 border-b border-border">
						<div className="flex items-start justify-between">
							<div>
								<h1 className="text-2xl font-semibold text-foreground">CovenantAI — Analysis</h1>
								<p className="text-sm text-muted-foreground mt-2 max-w-2xl">A clear, executive-ready summary and detailed breakdown of risks, negotiation points, and flags.</p>
							</div>
							<div className="flex items-center gap-3">
								<button onClick={handleDownloadPDF} disabled={isDownloading} className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-background border border-border text-foreground">
									{isDownloading ? <Loader2 className="animate-spin" size={16} /> : <Download size={16} />}
									<span className="hidden sm:inline">Export</span>
								</button>
								<div className="relative download-menu-anchor">
									<button onClick={() => setShowDownloadMenu(s => !s)} className="p-2 rounded-md bg-background border border-border">
										<ChevronDown size={14} />
									</button>
									{showDownloadMenu && (
										<div className="absolute right-0 mt-2 w-44 bg-card rounded-lg border border-border shadow z-40">
											<button onClick={async () => { setShowDownloadMenu(false); await handleDownloadPDF(); }} className="w-full text-left px-3 py-2 hover:bg-muted/50">Download PDF</button>
											<button onClick={async () => { setShowDownloadMenu(false); /* bundle */ await handleDownloadPDF(); }} className="w-full text-left px-3 py-2 hover:bg-muted/50">Download Bundle</button>
										</div>
									)}
								</div>
							</div>
						</div>
					</div>

					<div className="flex-1 min-h-0 flex">
						{/* Left nav */}
						<aside className="w-64 border-r border-border p-6 overflow-auto">
							<div className="space-y-4">
								{sections.map((s) => {
									const Icon = s.icon;
									const active = activeSection === s.key;
									return (
										<button key={s.key} onClick={() => setActiveSection(s.key)} className={`w-full text-left flex items-center gap-3 p-3 rounded-lg ${active ? 'bg-card/10' : 'hover:bg-card/5'}`}>
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
							<div className="max-w-none bg-card rounded-xl shadow-sm border border-border h-full">
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

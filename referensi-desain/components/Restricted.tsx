import React, { useEffect, useState } from "react";
import {
  ArrowRight,
  Check,
  Copy,
  Lock,
  MessageCircle,
  ShieldX,
  Sparkles,
  Terminal,
  User,
} from "lucide-react";

type Props = { reason?: string };

const steps = [
  { title: "Buka bot dan jalankan perintah", icon: Terminal },
  { title: "Kirim pesan ke admin", icon: MessageCircle },
  { title: "Berikan User ID Anda", icon: User },
];

export const Restricted: React.FC<Props> = ({ reason }) => {
  const [activeStep, setActiveStep] = useState(0);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    const id = setInterval(() => {
      setActiveStep((prev) => (prev + 1) % steps.length);
    }, 3000);
    return () => clearInterval(id);
  }, []);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText("/getid");
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // ignore clipboard failure
    }
  };

  return (
    <div className="relative z-10 max-w-xl mx-auto px-4 py-6 space-y-6">
      {/* Header */}
      <div className="text-center space-y-3">
        <div className="mx-auto w-20 h-20 rounded-full border border-slate-800 bg-gradient-to-br from-slate-950 to-black flex items-center justify-center shadow-[0_20px_40px_rgba(0,0,0,0.7)] transition-transform duration-300 hover:scale-105 hover:shadow-[0_26px_60px_rgba(0,0,0,0.9)]">
          <Lock className="text-slate-200" size={32} />
        </div>
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight">Akses terbatas</h1>
          <p className="text-sm text-slate-400">Dashboard hanya untuk admin grup.</p>
        </div>
      </div>

      {/* Alert */}
      <div className="glass-panel rounded-2xl p-4 border border-red-900/40 bg-red-950/40 flex items-start space-x-3">
        <div className="mt-1">
          <ShieldX className="text-red-400" size={18} />
        </div>
        <div className="space-y-1">
          <h3 className="text-sm font-semibold text-red-300">Anda tidak memiliki akses ke dashboard ini.</h3>
          <p className="text-xs text-red-400/80">
            {reason || "Detail: User ID Anda belum terdaftar sebagai admin grup."}
          </p>
        </div>
      </div>

      {/* Main card */}
      <div className="glass-panel rounded-2xl p-5 border border-slate-800 bg-slate-950/70 shadow-[0_25px_60px_rgba(0,0,0,0.7)] relative overflow-hidden">
        <div className="absolute inset-0 pointer-events-none">
          <div className="absolute -top-24 -right-24 w-56 h-56 rounded-full bg-slate-900/40 blur-3xl" />
          <div className="absolute -bottom-24 -left-24 w-56 h-56 rounded-full bg-purple-900/30 blur-3xl" />
        </div>

        <div className="relative z-10 space-y-5">
          <div className="flex items-center space-x-2">
            <Sparkles className="text-yellow-300" size={18} />
            <h3 className="text-lg font-semibold text-white">Ingin akses sebagai admin?</h3>
          </div>

          {/* Step indicator */}
          <div className="space-y-3">
            <div className="flex items-center justify-between relative">
              <div className="absolute left-6 right-6 h-px bg-slate-800" />
              {steps.map((step, idx) => {
                const Icon = step.icon;
                const isActive = idx === activeStep;
                return (
                  <button
                    key={step.title}
                    type="button"
                    onClick={() => setActiveStep(idx)}
                    className="relative z-10 flex flex-col items-center flex-1"
                  >
                    <div
                      className={`w-12 h-12 rounded-full border flex items-center justify-center transition-all duration-300 ${
                        isActive
                          ? "border-slate-300 bg-slate-900 text-white shadow-[0_0_20px_rgba(148,163,184,0.4)]"
                          : "border-slate-700 bg-slate-950 text-slate-500"
                      }`}
                    >
                      <Icon size={18} />
                    </div>
                  </button>
                );
              })}
            </div>
            <div className="mt-2 text-center bg-slate-950/80 border border-slate-800 rounded-xl py-2 px-3">
              <p className="text-xs text-slate-500">Langkah {activeStep + 1}</p>
              <p className="text-sm text-slate-200 font-medium mt-0.5">
                {steps[activeStep].title}
              </p>
            </div>
          </div>

          {/* Instructions */}
          <div className="space-y-3">
            <div className="flex items-center justify-between rounded-xl border border-slate-800 bg-slate-950/80 px-3 py-2.5">
              <div className="flex items-center space-x-3">
                <Terminal className="text-purple-400" size={18} />
                <div>
                  <p className="text-xs text-slate-400">Jalankan perintah ini di bot:</p>
                  <p className="mt-1 inline-flex items-center rounded bg-black/60 px-2 py-1 text-xs font-mono text-slate-100 border border-slate-700">
                    /getid
                  </p>
                </div>
              </div>
              <button
                type="button"
                onClick={handleCopy}
                className={`ml-2 flex h-9 w-9 items-center justify-center rounded-lg border text-slate-300 transition ${
                  copied
                    ? "bg-emerald-600/20 border-emerald-500 text-emerald-300"
                    : "bg-slate-900 border-slate-700 hover:bg-slate-800"
                }`}
              >
                {copied ? <Check size={16} /> : <Copy size={16} />}
              </button>
            </div>

            <div className="flex items-center justify-between rounded-xl border border-slate-800 bg-slate-950/80 px-3 py-2.5">
              <div className="flex items-center space-x-3">
                <MessageCircle className="text-sky-400" size={18} />
                <div>
                  <p className="text-xs text-slate-400">
                    Kirim pesan ke admin dengan menyertakan User ID Anda.
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Contact button */}
          <a
            href="https://t.me/coksadu"
            className="relative mt-2 inline-flex w-full items-center justify-center overflow-hidden rounded-xl border border-slate-700 bg-gradient-to-r from-slate-950 via-black to-slate-900 px-4 py-3 text-sm font-semibold text-white shadow-[0_10px_30px_rgba(0,0,0,0.7)]"
          >
            <span className="absolute inset-0 bg-gradient-to-r from-white/10 via-white/0 to-white/10 -translate-x-full animate-[shimmer_1.4s_infinite]" />
            <span className="relative flex items-center">
              <MessageCircle size={18} className="mr-2" />
              Hubungi @coksadu
              <ArrowRight size={16} className="ml-1" />
            </span>
          </a>
        </div>
      </div>
    </div>
  );
};

// tailwind keyframes helper (used by className above):
// @keyframes shimmer { 100% { transform: translateX(100%); } }

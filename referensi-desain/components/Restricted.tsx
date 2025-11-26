import React from "react";

type Props = { reason?: string };

export const Restricted: React.FC<Props> = ({ reason }) => {
  return (
    <div className="p-6 space-y-6">
      <div className="glass-panel rounded-2xl p-5 border border-red-500/30 bg-red-500/5">
        <p className="text-sm text-red-300 font-semibold">Dashboard hanya untuk admin grup.</p>
        {reason && <p className="text-xs text-red-200 mt-1 opacity-80">Detail: {reason}</p>}
      </div>

      <div className="glass-panel rounded-2xl p-5 border border-slate-800">
        <h3 className="text-white font-bold text-lg mb-2">Mau akses sebagai admin?</h3>
        <ol className="list-decimal list-inside text-sm text-slate-300 space-y-1">
          <li>Buka bot dan jalankan perintah <span className="font-mono text-primary-200">/getid</span> untuk melihat User ID Anda.</li>
          <li>Kirim pesan ke admin melalui tombol di bawah.</li>
          <li>Berikan User ID Anda agar ditambahkan sebagai admin grup.</li>
        </ol>

        <a
          href="https://t.me/coksadu"
          className="mt-4 inline-flex items-center justify-center px-4 py-2 rounded-xl bg-primary-600 text-white font-semibold w-full shadow-lg hover:bg-primary-500 transition-colors"
        >
          Hubungi @coksadu
        </a>
      </div>
    </div>
  );
};

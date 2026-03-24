"use client";

import { useEffect, useState, useCallback } from "react";
import Sidebar from "@/components/Sidebar";
import TradeTable from "@/components/TradeTable";
import PnlBarChart from "@/components/PnlBarChart";
import { fetchJSON, type Trade } from "@/lib/api";
import { Download } from "lucide-react";

type RiskLevel = "low" | "medium" | "high";
const riskColors: Record<string, string> = { low: "#4da6ff", medium: "#e8c300", high: "#00e87b" };

const pnlFmt = (v: number) =>
  `₹${v >= 0 ? "+" : ""}${v.toLocaleString("en-IN", { maximumFractionDigits: 0 })}`;

export default function TradesPage() {
  const [risk, setRisk] = useState<RiskLevel>("high");
  const [trades, setTrades] = useState<Trade[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<"ALL" | "CALL" | "PUT" | "WIN" | "LOSS" | "RL_EXIT">("ALL");

  const load = useCallback(async (r: RiskLevel) => {
    setLoading(true);
    try {
      const data = await fetchJSON<Trade[]>(`/api/trades/history?risk=${r}`);
      setTrades(Array.isArray(data) ? data : []);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(risk); }, [risk, load]);

  const filtered = trades.filter(t => {
    if (filter === "ALL") return true;
    if (filter === "CALL" || filter === "PUT") return t.direction === filter;
    if (filter === "WIN") return t.pnl > 0;
    if (filter === "LOSS") return t.pnl <= 0;
    if (filter === "RL_EXIT") return t.result === "RL_EXIT" || t.result === "DQN_EXIT";
    return true;
  });

  const totalPnl = filtered.reduce((s, t) => s + t.pnl, 0);
  const wins = filtered.filter(t => t.pnl > 0);
  const losses = filtered.filter(t => t.pnl <= 0);
  const winRate = filtered.length > 0 ? (wins.length / filtered.length * 100).toFixed(1) : "--";

  const strategies = Array.from(new Set(trades.map(t => t.strategy)));
  const byStrategy = strategies.map(s => ({
    strategy: s,
    trades: trades.filter(t => t.strategy === s).length,
    pnl: trades.filter(t => t.strategy === s).reduce((sum, t) => sum + t.pnl, 0),
    wr: trades.filter(t => t.strategy === s).length > 0
      ? (trades.filter(t => t.strategy === s && t.pnl > 0).length / trades.filter(t => t.strategy === s).length * 100).toFixed(0)
      : "0",
  }));

  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <main className="flex-1 p-5 overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-5">
          <div>
            <h1 className="text-sm font-bold uppercase tracking-wider" style={{ color: '#00e87b' }}>Trade History</h1>
            <p className="text-[10px] mt-0.5" style={{ color: '#3d4450' }}>ALL COMPLETED TRADES FROM BACKTEST</p>
          </div>
          <a
            href={`http://localhost:5050/api/trades/history?risk=${risk}`}
            download={`trades_${risk}.json`}
            className="t-btn flex items-center gap-1.5"
          >
            <Download className="w-3 h-3" /> EXPORT
          </a>
        </div>

        {/* Risk tabs */}
        <div className="flex gap-[1px] mb-4">
          {(["low", "medium", "high"] as RiskLevel[]).map(r => (
            <button key={r} onClick={() => setRisk(r)}
              className="px-4 py-[6px] text-[10px] font-semibold tracking-wider uppercase transition-all"
              style={{
                background: risk === r ? riskColors[r] : '#181c24',
                color: risk === r ? '#000' : '#5a6270',
                border: `1px solid ${risk === r ? riskColors[r] : '#252a33'}`,
              }}>
              {r} Risk
            </button>
          ))}
        </div>

        {/* Summary stats */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-[1px] mb-4">
          {[
            { label: "Total Trades", value: filtered.length, color: '#c8cdd5' },
            { label: "Total P&L", value: pnlFmt(totalPnl), color: totalPnl >= 0 ? '#00e87b' : '#ff3e3e' },
            { label: "Win Rate", value: `${winRate}%`, color: '#c8cdd5' },
            { label: "Winners", value: wins.length, color: '#00e87b' },
            { label: "Losers", value: losses.length, color: '#ff3e3e' },
          ].map(({ label, value, color }) => (
            <div key={label} className="t-panel p-3">
              <p className="text-[9px] uppercase tracking-[1.5px] mb-1" style={{ color: '#5a6270' }}>{label}</p>
              <p className="text-xl font-bold" style={{ color }}>{value}</p>
            </div>
          ))}
        </div>

        {/* Filter buttons */}
        <div className="flex gap-[1px] mb-4">
          {(["ALL", "CALL", "PUT", "WIN", "LOSS", "RL_EXIT"] as const).map(f => (
            <button key={f} onClick={() => setFilter(f)}
              className="px-3 py-[5px] text-[10px] font-semibold tracking-wider transition-all"
              style={{
                background: filter === f ? '#252a33' : '#181c24',
                color: filter === f ? '#c8cdd5' : '#3d4450',
                border: `1px solid ${filter === f ? '#333a45' : '#252a33'}`,
              }}>
              {f}
            </button>
          ))}
        </div>

        {/* P&L bar chart */}
        {filtered.length > 0 && (
          <div className="t-panel p-4 mb-4">
            <h3 className="text-[11px] font-semibold mb-3 uppercase tracking-wider" style={{ color: '#5a6270' }}>Per-Trade P&L</h3>
            <PnlBarChart trades={filtered} />
          </div>
        )}

        {/* Strategy breakdown */}
        {byStrategy.length > 0 && (
          <div className="t-panel p-4 mb-4">
            <h3 className="text-[11px] font-semibold mb-3 uppercase tracking-wider" style={{ color: '#5a6270' }}>Strategy Breakdown</h3>
            <table>
              <thead>
                <tr>
                  {["Strategy", "Trades", "P&L", "Win Rate"].map(h => (
                    <th key={h}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {byStrategy.map(s => (
                  <tr key={s.strategy}>
                    <td style={{ color: '#c8cdd5' }}>{s.strategy?.replace(/_/g, " ")}</td>
                    <td>{s.trades}</td>
                    <td style={{ color: s.pnl >= 0 ? '#00e87b' : '#ff3e3e', fontWeight: 600 }}>{pnlFmt(s.pnl)}</td>
                    <td>{s.wr}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Full trade table */}
        <div className="t-panel p-4">
          <h3 className="text-[11px] font-semibold mb-3 uppercase tracking-wider" style={{ color: '#5a6270' }}>
            All Trades {filter !== "ALL" ? `(${filter})` : ""} — {filtered.length} records
          </h3>
          {loading ? (
            <div className="h-32 flex items-center justify-center text-[11px]" style={{ color: '#3d4450' }}>LOADING...</div>
          ) : (
            <TradeTable trades={filtered} />
          )}
        </div>
      </main>
    </div>
  );
}

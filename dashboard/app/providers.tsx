"use client";

import { TradingModeProvider, useTradingMode } from "@/contexts/TradingModeContext";
import RetroDialog from "@/components/RetroDialog";

function GlobalDialogs() {
  const { showDialog, setShowDialog, dialogError } = useTradingMode();
  return (
    <RetroDialog
      open={showDialog}
      onClose={() => setShowDialog(false)}
      title="CONFIGURATION ERROR"
      message={dialogError ?? ""}
      type="error"
    />
  );
}

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <TradingModeProvider>
      {children}
      <GlobalDialogs />
    </TradingModeProvider>
  );
}

import type { Metadata } from "next";
import "./globals.css";
import Sidebar from "@/components/Sidebar";
import Topbar from "@/components/Topbar";

export const metadata: Metadata = {
  title: "GridPulse · Energy Time-Series Dashboard",
  description:
    "Forecasting, imputation and anomaly detection on electricity transformer data with RNN/LSTM/GRU models.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <Sidebar />
        <div className="ml-60 flex min-h-screen flex-col">
          <Topbar />
          <main className="flex-1 px-6 py-6">{children}</main>
        </div>
      </body>
    </html>
  );
}

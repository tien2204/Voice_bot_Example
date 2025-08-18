import Topbar from "@/components/Topbar/Topbar";
export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <>
      <Topbar />
      <main className="flex-grow overflow-auto">{children}</main>
    </>
  );
}

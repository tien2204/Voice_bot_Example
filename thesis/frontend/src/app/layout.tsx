import Footbar from "@/components/Footbar/Footbar";
import Topbar from "@/components/Topbar/Topbar";
import { AuthProvider, useAuth } from "@/contexts/AuthContext";
import "@livekit/components-styles";
import { Metadata } from "next";
import { Montserrat, Public_Sans } from "next/font/google";
import "./layout.css";

const publicSans400 = Public_Sans({
  weight: "400",
  subsets: ["latin"],
});
const montserrat = Montserrat({
  weight: ["400", "700"],
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Voice Assistant",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`min-h-screen ${montserrat.className}`}>
      <body className="min-h-screen flex flex-col">
        <AuthProvider>
          <main className="flex-grow flex flex-col">{children}</main>
          <Footbar />
        </AuthProvider>
      </body>
    </html>
  );
}

// function AppContent() {
//   const { isAuthenticated, user, logout } = useAuth();

//   if (!isAuthenticated || !user) {
//     return <LoginComponent />;
//   }

//   return (
//     <div style={styles.appContainer}>
//       <header style={styles.header}>
//         <h1>Avatar Creator</h1>
//         <div style={styles.userInfo}>Logged in as: {user.username} ({user.email}) <button onClick={logout}>Logout</button></div>
//       </header>
//       <AvatarCreatorComponent onAvatarExported={(url) => console.log('App: Avatar Exported', url)} />
//     </div>
//   );
// }

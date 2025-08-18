"use client";

import { SparkleDiv } from "@/components/Sparkle/SparkleDiv";
import { BasicHeadDisplay } from "@/components/TalkingHead/BasicHeadDisplay";
import { useAuth } from "@/contexts/AuthContext";
import Image from "next/image";
import { useRouter } from "next/navigation";
import React, { FormEvent, useEffect, useState } from "react";

const LogoWith3DModel = React.memo(function LogoWith3DModel() {
  const [logoMode, setLogoMode] = useState<"img" | "3d">("img");
  return (
    <div className="mx-auto w-48 aspect-square sm:w-56 md:w-64">
      <Image
        src="/favicon.png"
        alt="SehrMude Logo"
        width={300}
        height={300}
        className={`w-full h-full object-contain ${logoMode === "img" ? "" : "hidden"}`}
      />
      <div
        // Apply Tailwind classes for flex layout and visibility
        className={`w-full h-full flex flex-col items-center justify-center text-white ${logoMode === "3d" ? "" : "hidden"}`}
      >
        <div
          // This div takes full space of its parent, rounded, with overflow hidden
          className="w-full h-full rounded-full overflow-hidden"
        >
          <BasicHeadDisplay
            initialMood="neutral"
            initialPose="straight"
            lookAtCamera={true} // Corrected prop name
            initOptions={{
              cameraView: "head",
              modelBgColor: "#1f2937", // Matches bg-gray-800 (form card background)
            }}
            onLoaded={() => {
              if (logoMode !== "3d") {
                setLogoMode("3d");
              }
            }}
          />
        </div>
      </div>
    </div>
  );
});
export default function LoginPage() {
  const { login, isAuthenticated, isLoading, error } = useAuth();
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  useEffect(() => {
    if (isAuthenticated) {
      router.push("/avatar/chat");
    }
  }, [isAuthenticated, router]);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    await login(email, password);
  };

  if (isLoading && !isAuthenticated) {
    // Show loading indicator if auth state is being determined and user is not yet authenticated
    return (
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          height: "100vh",
          backgroundColor: "#111827",
          color: "white",
        }}
      >
        <div style={{ fontSize: "1.25rem" }}>Loading...</div>
      </div>
    );
  }

  // If already authenticated, the useEffect should have redirected.
  // This is a fallback or for when isLoading is false but isAuthenticated becomes true rapidly.
  if (isAuthenticated) {
    return <div>Redirecting...</div>;
  }

  return (
    <div className="flex flex-col  items-center justify-center bg-gray-900 p-4 flex-grow">
      {/* Adjusted padding for better scaling on small screens */}
      <div className="w-full max-w-md p-6 sm:p-8 space-y-8 bg-gray-800 shadow-xl rounded-xl">
        <div className="text-center topbar-logo">
          <LogoWith3DModel />
          <h1 className="mt-6 text-3xl font-extrabold text-gray-100">
            Sign in to your{" "}
            <SparkleDiv>
              <span className="text-sky-400">{process.env.NEXT_PUBLIC_APP_NAME}</span>
            </SparkleDiv>{" "}
            account
          </h1>
        </div>
        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label htmlFor="email" className="block text-sm font-medium text-gray-300">
              Email address
            </label>
            <input
              id="email"
              type="email"
              autoComplete="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              className="mt-1 appearance-none block w-full px-3 py-2 border border-gray-600 bg-gray-700 text-gray-100 rounded-lg shadow-sm placeholder-gray-400 focus:outline-none focus:ring-gray-500 focus:border-gray-500 sm:text-sm"
              placeholder="you@example.com"
            />
          </div>
          <div>
            <label htmlFor="password" className="block text-sm font-medium text-gray-300">
              Password
            </label>
            <input
              id="password"
              type="password"
              autoComplete="current-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              className="mt-1 appearance-none block w-full px-3 py-2 border border-gray-600 bg-gray-700 text-gray-100 rounded-lg shadow-sm placeholder-gray-400 focus:outline-none focus:ring-gray-500 focus:border-gray-500 sm:text-sm"
              placeholder="••••••••"
            />
          </div>
          {error && <p className="text-xs text-red-400 text-center">{error}</p>}
          <input
            type="submit"
            disabled={isLoading}
            className={`w-full flex justify-center py-3 px-4 border border-transparent rounded-lg shadow-sm text-sm font-medium text-white transition duration-150 ease-in-out
                        ${isLoading ? "bg-gray-600 cursor-not-allowed" : "bg-gray-700 hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-800 focus:ring-gray-500 cursor-pointer"}`}
            value={isLoading ? "Logging in..." : "Login"}
          />
          <div className="text-center">
            <button
              type="button"
              onClick={() => router.push("/register")} // Assuming you have a /register route
              className="w-full flex justify-center py-3 px-4 border border-transparent rounded-lg shadow-sm text-sm font-medium text-sky-400 hover:text-sky-300 bg-transparent hover:bg-gray-700 focus:outline-none cursor-pointer mt-4"
            >
              Don't have an account? Register
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

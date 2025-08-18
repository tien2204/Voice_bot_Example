"use client";

import { SparkleDiv } from "@/components/Sparkle/SparkleDiv";
import { useAuth } from "@/contexts/AuthContext";
import Image from "next/image";
import { useRouter } from "next/navigation";
import { FormEvent, useEffect, useState } from "react";

export default function RegisterPage() {
  const { register, isAuthenticated, isLoading, error, clearError } = useAuth();
  const router = useRouter();
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  useEffect(() => {
    if (isAuthenticated) {
      router.push("/avatar/chat"); // Or wherever authenticated users should go
    }
    // Clear any previous errors when the component mounts or isAuthenticated changes
    return () => {
      clearError();
    };
  }, [isAuthenticated, router, clearError]);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    clearError(); // Clear previous errors before a new attempt
    const success = await register(username, email, password);
    if (success) {
      router.push("/verify-email-sent"); // Redirect to a page informing the user to check their email
    }
    // If not successful, the error state in AuthContext will be set and displayed
  };

  if (isLoading && !isAuthenticated) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100vh', backgroundColor: '#111827', color: 'white' }}>
        <div style={{fontSize: '1.25rem'}}>Loading...</div>
      </div>
    );
  }

  if (isAuthenticated) {
    // This case should ideally be handled by the useEffect redirect
    return <div>Redirecting...</div>;
  }

  return (
    <div className="flex flex-col  items-center justify-center bg-gray-900 p-4 flex-grow">
      <div className="w-full max-w-md p-8 space-y-8 bg-gray-800 shadow-xl rounded-xl">
        <div className="text-center">
          <div className="topbar-logo mx-auto w-48 aspect-square sm:w-56 md:w-64">
            <Image src="/favicon.png" alt="SehrMude Logo" width={300} height={300} />
          </div>
          <h1 className="mt-6 text-3xl font-extrabold text-gray-100">
            Create your{" "}
            <SparkleDiv>
              <span className="text-sky-400">{process.env.NEXT_PUBLIC_APP_NAME}</span>
            </SparkleDiv>{" "}
            account
          </h1>
        </div>
        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label htmlFor="username" className="block text-sm font-medium text-gray-300">
              Username
            </label>
            <input
              id="username"
              type="text"
              autoComplete="username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
              className="mt-1 appearance-none block w-full px-3 py-2 border border-gray-600 bg-gray-700 text-gray-100 rounded-lg shadow-sm placeholder-gray-400 focus:outline-none focus:ring-gray-500 focus:border-gray-500 sm:text-sm"
              placeholder="Your username"
            />
          </div>
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
              autoComplete="new-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              className="mt-1 appearance-none block w-full px-3 py-2 border border-gray-600 bg-gray-700 text-gray-100 rounded-lg shadow-sm placeholder-gray-400 focus:outline-none focus:ring-gray-500 focus:border-gray-500 sm:text-sm"
              placeholder="Password of minimum 8 characters"
            />
          </div>
          {error && <p className="text-xs text-red-400 text-center">{error}</p>}
          <input
            type="submit"
            disabled={isLoading}
            className={`w-full flex justify-center py-3 px-4 border border-transparent rounded-lg shadow-sm text-sm font-medium text-white transition duration-150 ease-in-out
                        ${isLoading ? "bg-gray-600 cursor-not-allowed" : "bg-sky-500 hover:bg-sky-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-800 focus:ring-sky-400 cursor-pointer"}`}
            value={isLoading ? "Registering..." : "Register"}
          />
          <div className="text-center">
            <button
              type="button"
              onClick={() => router.push("/login")}
              className="w-full flex justify-center py-3 px-4 border border-transparent rounded-lg shadow-sm text-sm font-medium text-sky-400 hover:text-sky-300 bg-transparent hover:bg-gray-700 focus:outline-none cursor-pointer mt-4"
            >
              Already have an account? Login
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
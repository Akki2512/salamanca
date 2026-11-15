#include "RemoteHandsReceiver.h"

#include "Common/UdpSocketBuilder.h"
#include "Components/ArrowComponent.h"
#include "Components/PoseableMeshComponent.h"
#include "Components/SceneCaptureComponent2D.h"
#include "Components/SceneComponent.h"
#include "Containers/Ticker.h"
#include "Dom/JsonObject.h"
#include "DrawDebugHelpers.h"
#include "Engine/Engine.h"
#include "Engine/TextureRenderTarget2D.h"
#include "Engine/World.h"
#include "HttpPath.h"
#include "HttpServerModule.h"
#include "HttpServerResponse.h"
#include "IHttpRouter.h"
#include "ImageUtils.h"
#include "TextureResource.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "SocketSubsystem.h"
#include "Sockets.h"

#if WITH_EDITOR
#include "Editor.h"
#include "Editor/EditorPerformanceSettings.h"
#include "LevelEditorViewport.h"
#endif

DEFINE_LOG_CATEGORY_STATIC(LogRemoteHands, Log, All);

namespace
{
	// The 28 joints of the packet, as named in the G1 URDF (g1_body29_hand14).
	const TCHAR* const GJointNames[] = {
		TEXT("left_shoulder_pitch_joint"), TEXT("left_shoulder_roll_joint"), TEXT("left_shoulder_yaw_joint"),
		TEXT("left_elbow_joint"), TEXT("left_wrist_roll_joint"), TEXT("left_wrist_pitch_joint"), TEXT("left_wrist_yaw_joint"),
		TEXT("right_shoulder_pitch_joint"), TEXT("right_shoulder_roll_joint"), TEXT("right_shoulder_yaw_joint"),
		TEXT("right_elbow_joint"), TEXT("right_wrist_roll_joint"), TEXT("right_wrist_pitch_joint"), TEXT("right_wrist_yaw_joint"),
		TEXT("left_hand_thumb_0_joint"), TEXT("left_hand_thumb_1_joint"), TEXT("left_hand_thumb_2_joint"),
		TEXT("left_hand_middle_0_joint"), TEXT("left_hand_middle_1_joint"), TEXT("left_hand_index_0_joint"), TEXT("left_hand_index_1_joint"),
		TEXT("right_hand_thumb_0_joint"), TEXT("right_hand_thumb_1_joint"), TEXT("right_hand_thumb_2_joint"),
		TEXT("right_hand_index_0_joint"), TEXT("right_hand_index_1_joint"), TEXT("right_hand_middle_0_joint"), TEXT("right_hand_middle_1_joint"),
		// legs and waist: sent only by a sender whose robot walks (ext.joints)
		TEXT("left_hip_pitch_joint"), TEXT("left_hip_roll_joint"), TEXT("left_hip_yaw_joint"), TEXT("left_knee_joint"),
		TEXT("left_ankle_pitch_joint"), TEXT("left_ankle_roll_joint"),
		TEXT("right_hip_pitch_joint"), TEXT("right_hip_roll_joint"), TEXT("right_hip_yaw_joint"), TEXT("right_knee_joint"),
		TEXT("right_ankle_pitch_joint"), TEXT("right_ankle_roll_joint"),
		TEXT("waist_yaw_joint"), TEXT("waist_roll_joint"), TEXT("waist_pitch_joint"),
	};

	// Joint axis from the URDF (right-handed: x forward, y left, z up), by joint name:
	//   pitch joints, elbows, knees and thumb_0 turn about y; roll joints about x; yaw joints and the other finger joints about z.
	FVector UrdfAxis(const FString& Joint)
	{
		if (Joint.Contains(TEXT("_roll_")))
		{
			return FVector(1, 0, 0);
		}
		if (Joint.Contains(TEXT("_pitch_")) || Joint.Contains(TEXT("_elbow_")) || Joint.Contains(TEXT("_knee_")) || Joint.Contains(TEXT("_thumb_0_")))
		{
			return FVector(0, 1, 0);
		}
		return FVector(0, 0, 1);
	}

	// Unreal is left-handed (y to the right). Mirroring y turns a rotation by +a about (x, y, z) into a rotation
	// by -a about (x, -y, z), which is +a about (-x, y, -z). That is the default axis used here; it is right for a
	// rig whose parts keep the URDF's frames with y mirrored. Anything else: fix Axis / Scale per joint.
	FVector DefaultUnrealAxis(const FString& Joint)
	{
		const FVector A = UrdfAxis(Joint);
		return FVector(-A.X, A.Y, -A.Z);
	}

	FVector RobotMetresToLocalCm(const FVector& P)
	{
		return FVector(P.X * 100.0, -P.Y * 100.0, P.Z * 100.0);
	}

	// A pose {pos:[x,y,z], quat:[w,x,y,z]} in the sender's right-handed frame -> an Unreal transform in that same
	// frame mirrored in y: position (100 x, -100 y, 100 z) cm, quaternion (X, Y, Z, W) = (-x, y, -z, w).
	bool ReadPose(const TSharedPtr<FJsonObject>& Obj, FTransform& Out)
	{
		const TArray<TSharedPtr<FJsonValue>>* P = nullptr;
		const TArray<TSharedPtr<FJsonValue>>* Q = nullptr;
		if (!Obj.IsValid() || !Obj->TryGetArrayField(TEXT("pos"), P) || P->Num() < 3)
		{
			return false;
		}
		FQuat Rot = FQuat::Identity;
		if (Obj->TryGetArrayField(TEXT("quat"), Q) && Q->Num() >= 4)
		{
			Rot = FQuat(-(*Q)[1]->AsNumber(), (*Q)[2]->AsNumber(), -(*Q)[3]->AsNumber(), (*Q)[0]->AsNumber()).GetNormalized();
		}
		Out = FTransform(Rot, RobotMetresToLocalCm(FVector((*P)[0]->AsNumber(), (*P)[1]->AsNumber(), (*P)[2]->AsNumber())));
		return true;
	}

	bool ReadVec(const TArray<TSharedPtr<FJsonValue>>* Arr, FVector& Out)
	{
		if (Arr == nullptr || Arr->Num() < 3)
		{
			return false;
		}
		Out = FVector((*Arr)[0]->AsNumber(), (*Arr)[1]->AsNumber(), (*Arr)[2]->AsNumber());
		return true;
	}

	// Scene component named Name on Root or on any actor attached to it (some URDF importers make one actor per link).
	USceneComponent* FindPart(AActor* Root, const FString& Name)
	{
		if (Root == nullptr)
		{
			return nullptr;
		}
		TArray<AActor*> Actors;
		Root->GetAttachedActors(Actors, true, true);
		Actors.Insert(Root, 0);
		for (AActor* A : Actors)
		{
			if (A == nullptr)
			{
				continue;
			}
			TInlineComponentArray<USceneComponent*> Comps(A);
			for (USceneComponent* C : Comps)
			{
				if (C != nullptr && C->GetName().Equals(Name, ESearchCase::IgnoreCase))
				{
					return C;
				}
			}
			if (A != Root && A->GetRootComponent() != nullptr &&
				(A->GetName().Equals(Name, ESearchCase::IgnoreCase) || A->GetActorNameOrLabel().Equals(Name, ESearchCase::IgnoreCase)))
			{
				return A->GetRootComponent();
			}
		}
		return nullptr;
	}
}

ARemoteHandsReceiver::ARemoteHandsReceiver()
{
	PrimaryActorTick.bCanEverTick = true;
	PrimaryActorTick.bStartWithTickEnabled = true;
	USceneComponent* Root = CreateDefaultSubobject<USceneComponent>(TEXT("RobotFrame"));
	SetRootComponent(Root);
	UArrowComponent* Arrow = CreateDefaultSubobject<UArrowComponent>(TEXT("Forward"));   // shows which way the robot faces
	Arrow->SetupAttachment(Root);
	Arrow->ArrowSize = 0.5f;
	Arrow->SetHiddenInGame(true);
}

FVector ARemoteHandsReceiver::SimToWorld(const FVector& SimMetres) const
{
	return SimWorldOrigin.TransformPosition(RobotMetresToLocalCm(SimMetres));
}

FVector ARemoteHandsReceiver::RobotToWorld(const FVector& RobotMetres) const
{
	return GetActorTransform().TransformPosition(RobotMetresToLocalCm(RobotMetres));
}

UPoseableMeshComponent* ARemoteHandsReceiver::FindPoseable() const
{
	return RobotActor != nullptr ? RobotActor->FindComponentByClass<UPoseableMeshComponent>() : nullptr;
}

void ARemoteHandsReceiver::AutoBindJoints()
{
	if (RobotActor == nullptr)
	{
		UE_LOG(LogRemoteHands, Warning, TEXT("Auto Bind Joints: set RobotActor first."));
		return;
	}
	Modify();
	UPoseableMeshComponent* Poseable = FindPoseable();
	int32 Found = 0;
	FString Missing;
	for (const TCHAR* JointChars : GJointNames)
	{
		const FString Joint(JointChars);
		const FString Link = Joint.Replace(TEXT("_joint"), TEXT("_link"));
		FRemoteHandsJointBinding B;
		if (const FRemoteHandsJointBinding* Old = Joints.Find(FName(*Joint)))
		{
			B = *Old;                                   // keep hand-tuned Axis / Scale / Offset
		}
		else
		{
			B.Axis = DefaultUnrealAxis(Joint);
		}
		bool bOk = false;
		// In the G1 URDF the waist pitch joint moves the link called torso_link.
		const FString Alias = Joint == TEXT("waist_pitch_joint") ? FString(TEXT("torso_link")) : Link;
		for (const FString& Candidate : { Joint, Link, Alias })
		{
			if (Poseable != nullptr)
			{
				const int32 Bone = Poseable->GetBoneIndex(FName(*Candidate));
				if (Bone != INDEX_NONE && Poseable->BoneSpaceTransforms.IsValidIndex(Bone))
				{
					B.Target = FName(*Candidate);
					B.bIsBone = true;
					B.Rest = Poseable->BoneSpaceTransforms[Bone].GetRotation();
					bOk = true;
					break;
				}
			}
			if (USceneComponent* Part = FindPart(RobotActor, Candidate))
			{
				B.Target = FName(*Candidate);
				B.bIsBone = false;
				B.Rest = Part->GetRelativeRotation().Quaternion();
				bOk = true;
				break;
			}
		}
		if (bOk)
		{
			Joints.Add(FName(*Joint), B);
			++Found;
		}
		else
		{
			Missing += Joint + TEXT(" ");
		}
	}
	UE_LOG(LogRemoteHands, Log, TEXT("Auto Bind Joints: %d of %d joints found on %s.%s%s"), Found,
		int32(UE_ARRAY_COUNT(GJointNames)), *RobotActor->GetActorNameOrLabel(),
		Missing.IsEmpty() ? TEXT("") : TEXT(" Not found (no component or bone with the joint's or link's name): "), *Missing);
}

// ---------------------------------------------------------------------------------------------------------
// The shared listener. One socket for the whole process, pumped by the core ticker (runs every engine frame,
// independent of world ticking and of which viewport is realtime).
namespace RemoteHandsLink
{
	TArray<TWeakObjectPtr<ARemoteHandsReceiver>> GReceivers;
	TMap<int32, FSocket*> GSockets;          // port -> socket
	double GNextOpenTryAt = 0.0;
	FTSTicker::FDelegateHandle GTicker;
	TArray<uint8> GBuffer;

	// ---- the robot's-eye picture: newest JPEG, served at /eyes.jpg
	FCriticalSection GEyeLock;
	TArray<uint8> GEyeJpeg;
	int32 GEyeSeq = 0;
	int32 GEyePort = 0;
	// Where the headset wearer's head points, from the viewer's requests (degrees: yaw to the left, pitch up),
	// and the direction the newest frame was rendered at, so the viewer can place it there.
	double GHeadAt = -1.0e9;
	float GHeadYaw = 0.f, GHeadPitch = 0.f;
	FString GEyeTag = TEXT("flat");
	double GNextEyeAt = 0.0;
	TArray<uint8> GPersonJpeg;          // the person's own view, newest frame
	int32 GPersonSeq = 0;
	double GNextPersonAt = 0.0;
	FHttpRouteHandle GPersonRoute;
	TSharedPtr<IHttpRouter> GEyeRouter;
	FHttpRouteHandle GEyeRoute;

	bool GViewportsPaused = false;
	double GStreamingAt = -1.0e9;       // when the eye picture was last being streamed
	bool GNoThrottleHooked = false;

	// Pause / resume the editor's level viewports (a named realtime override, so the user's own setting returns).
	void PauseEditorViewports(bool bPause)
	{
#if WITH_EDITOR
		if (bPause == GViewportsPaused || GEditor == nullptr)
		{
			return;
		}
		GViewportsPaused = bPause;
		for (FLevelEditorViewportClient* VC : GEditor->GetLevelViewportClients())
		{
			if (VC == nullptr)
			{
				continue;
			}
			if (bPause)
			{
				VC->AddRealtimeOverride(false, NSLOCTEXT("RemoteHands", "EyesStreaming", "Remote Hands: robot's eyes streaming"));
			}
			else
			{
				VC->RemoveRealtimeOverride(NSLOCTEXT("RemoteHands", "EyesStreaming", "Remote Hands: robot's eyes streaming"), false);
				VC->Invalidate();
			}
		}
		UE_LOG(LogRemoteHands, Log, TEXT("Editor viewports %s."), bPause ? TEXT("paused while the robot's eyes stream") : TEXT("resumed"));
#endif
	}

	void ServeEyes(int32 Port)
	{
		if (GEyePort == Port && GEyeRouter.IsValid())
		{
			return;
		}
		if (GEyeRouter.IsValid() && GEyeRoute.IsValid())
		{
			GEyeRouter->UnbindRoute(GEyeRoute);
		}
		GEyePort = Port;
		GEyeRouter = FHttpServerModule::Get().GetHttpRouter(Port, true);
		if (!GEyeRouter.IsValid())
		{
			UE_LOG(LogRemoteHands, Warning, TEXT("Could not serve the eye picture on port %d (in use?)."), Port);
			return;
		}
		GEyeRoute = GEyeRouter->BindRoute(FHttpPath(TEXT("/eyes.jpg")), EHttpServerRequestVerbs::VERB_GET,
			FHttpRequestHandler::CreateLambda([](const FHttpServerRequest& Request, const FHttpResultCallback& OnComplete)
			{
				if (Request.QueryParams.Contains(TEXT("vr")))        // a headset: which way is the wearer looking?
				{
					const FString* Y = Request.QueryParams.Find(TEXT("y"));
					const FString* P = Request.QueryParams.Find(TEXT("p"));
					GHeadYaw = Y != nullptr ? FMath::Clamp(FCString::Atof(**Y), -180.f, 180.f) : 0.f;
					GHeadPitch = P != nullptr ? FMath::Clamp(FCString::Atof(**P), -85.f, 85.f) : 0.f;
					GHeadAt = FPlatformTime::Seconds();
				}
				TArray<uint8> Bytes;
				int32 Seq = 0;
				FString Tag;
				{
					FScopeLock Lock(&GEyeLock);
					Bytes = GEyeJpeg;
					Seq = GEyeSeq;
					Tag = GEyeTag;
				}
				TUniquePtr<FHttpServerResponse> R = FHttpServerResponse::Create(MoveTemp(Bytes), TEXT("image/jpeg"));
				R->Code = Seq > 0 ? EHttpServerResponseCodes::Ok : EHttpServerResponseCodes::NoContent;
				R->Headers.Add(TEXT("Cache-Control"), { TEXT("no-store") });
				R->Headers.Add(TEXT("Access-Control-Allow-Origin"), { TEXT("*") });
				R->Headers.Add(TEXT("X-Frame"), { FString::FromInt(Seq) });
				R->Headers.Add(TEXT("X-Head"), { Tag });           // "yaw,pitch,horizontal fov,aspect" of this frame, or "flat"
				R->Headers.Add(TEXT("Access-Control-Expose-Headers"), { TEXT("X-Frame, X-Head") });
				OnComplete(MoveTemp(R));
				return true;
			}));
		GPersonRoute = GEyeRouter->BindRoute(FHttpPath(TEXT("/person.jpg")), EHttpServerRequestVerbs::VERB_GET,
			FHttpRequestHandler::CreateLambda([](const FHttpServerRequest& Request, const FHttpResultCallback& OnComplete)
			{
				TArray<uint8> Bytes;
				int32 Seq = 0;
				{
					FScopeLock Lock(&GEyeLock);
					Bytes = GPersonJpeg;
					Seq = GPersonSeq;
				}
				TUniquePtr<FHttpServerResponse> R = FHttpServerResponse::Create(MoveTemp(Bytes), TEXT("image/jpeg"));
				R->Code = Seq > 0 ? EHttpServerResponseCodes::Ok : EHttpServerResponseCodes::NoContent;
				R->Headers.Add(TEXT("Cache-Control"), { TEXT("no-store") });
				R->Headers.Add(TEXT("Access-Control-Allow-Origin"), { TEXT("*") });
				OnComplete(MoveTemp(R));
				return true;
			}));
		FHttpServerModule::Get().StartAllListeners();
		UE_LOG(LogRemoteHands, Log, TEXT("Serving the robot's-eye picture at http://127.0.0.1:%d/eyes.jpg"), Port);
	}

	void CloseSocket(int32 Port)
	{
		if (FSocket** S = GSockets.Find(Port))
		{
			if (*S != nullptr)
			{
				(*S)->Close();
				ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->DestroySocket(*S);
			}
			GSockets.Remove(Port);
		}
	}

	void CloseAll()
	{
		TArray<int32> Ports;
		GSockets.GetKeys(Ports);
		for (int32 P : Ports)
		{
			CloseSocket(P);
		}
	}

	bool Pump(float)
	{
		GReceivers.RemoveAll([](const TWeakObjectPtr<ARemoteHandsReceiver>& R) { return !R.IsValid(); });
		if (GReceivers.Num() == 0)
		{
			CloseAll();
			PauseEditorViewports(false);
			return true;
		}
		// Who gets the packets: receivers in a running Play / Simulate world if there is one, else the ones
		// in the level being edited.
		bool bGame = false;
		for (const TWeakObjectPtr<ARemoteHandsReceiver>& R : GReceivers)
		{
			const UWorld* W = R->GetWorld();
			bGame |= W != nullptr && W->IsGameWorld();
		}
		TArray<ARemoteHandsReceiver*> Targets;
		for (const TWeakObjectPtr<ARemoteHandsReceiver>& R : GReceivers)
		{
			const UWorld* W = R->GetWorld();
			if (W == nullptr || R->IsTemplate())
			{
				continue;
			}
			if (bGame ? W->IsGameWorld() : (W->WorldType == EWorldType::Editor && R->bRunInEditor))
			{
				Targets.Add(R.Get());
			}
		}
		const double Now = FPlatformTime::Seconds();
		TArray<int32> Want;
		if (Targets.Num() > 0)
		{
			Want.AddUnique(Targets[0]->Port);
			for (int32 P : Targets[0]->ExtraPorts)
			{
				if (P > 0 && P < 65536)
				{
					Want.AddUnique(P);
				}
			}
			TArray<int32> Open;
			GSockets.GetKeys(Open);
			for (int32 P : Open)
			{
				if (!Want.Contains(P))
				{
					CloseSocket(P);
				}
			}
			if (Now >= GNextOpenTryAt)
			{
				for (int32 P : Want)
				{
					if (GSockets.Contains(P))
					{
						continue;
					}
					GNextOpenTryAt = Now + 2.0;
					FSocket* S = FUdpSocketBuilder(TEXT("RemoteHandsLink")).AsNonBlocking().AsReusable().BoundToPort(P).WithReceiveBufferSize(1 << 20).Build();
					if (S == nullptr)
					{
						UE_LOG(LogRemoteHands, Warning, TEXT("Could not open UDP port %d (in use?). Trying again in 2 s."), P);
					}
					else
					{
						GSockets.Add(P, S);
						UE_LOG(LogRemoteHands, Log, TEXT("Listening for a robot on UDP port %d."), P);
					}
				}
			}
		}
		bool bGot = false;
		int32 LastSize = 0;
		for (const TPair<int32, FSocket*>& KV : GSockets)
		{
			// Newest wins: read everything that has arrived and apply only the last datagram.
			uint32 Pending = 0;
			TArray<uint8> Datagram;
			while (KV.Value->HasPendingData(Pending) && Pending > 0)
			{
				Datagram.SetNumUninitialized(FMath::Min<uint32>(Pending, 65507u));
				int32 Read = 0;
				if (!KV.Value->Recv(Datagram.GetData(), Datagram.Num(), Read) || Read <= 0)
				{
					break;
				}
				GBuffer = Datagram;
				LastSize = Read;
			}
		}
		if (LastSize > 0)
		{
			const FUTF8ToTCHAR Text(reinterpret_cast<const ANSICHAR*>(GBuffer.GetData()), LastSize);
			const FString Json(Text.Length(), Text.Get());
			for (ARemoteHandsReceiver* R : Targets)
			{
				bGot |= R->ApplyPacket(Json);
			}
		}
		for (ARemoteHandsReceiver* R : Targets)
		{
			R->NoteLink(Now, bGot);
		}
		// The robot's eyes: one receiver renders its camera, at its own frame rate, after the pose was applied.
		bool bEyes = false, bPause = false;
		for (ARemoteHandsReceiver* R : Targets)
		{
			if (R->bStreamEyes && R->EyeCamera != nullptr)
			{
				bEyes = true;
				bPause = R->bPauseEditorViewports && !bGame;
				ServeEyes(R->EyeHttpPort);
				if (Now >= GNextEyeAt)
				{
					// keep the average rate even when the editor's own frame rate does not divide into it
					const double Period = 1.0 / FMath::Clamp(R->EyeFps, 1.f, 60.f);
					GNextEyeAt = FMath::Max(GNextEyeAt + Period, Now - Period);
					R->CaptureEyes();
				}
				if (R->PersonCamera != nullptr && Now >= GNextPersonAt)
				{
					const double Period = 1.0 / FMath::Clamp(R->PersonFps, 1.f, 30.f);
					GNextPersonAt = FMath::Max(GNextPersonAt + Period, Now - Period);
					R->CapturePerson();
				}
				break;
			}
		}
		PauseEditorViewports(bEyes && bPause);
#if WITH_EDITOR
		if (bEyes)
		{
			// "Use less CPU when in background" drops the whole editor to 3 frames a second when another window
			// is in front, which it always is while someone wears the headset. Off for as long as we stream.
			// The editor also drops to 3 frames a second when it is not in front and all its windows are hidden
			// (covered, minimised, or on another desktop), whatever that setting says. It asks these delegates
			// first: say "do not throttle" for as long as the eyes are streaming.
			GStreamingAt = FPlatformTime::Seconds();
			if (!GNoThrottleHooked && GEditor != nullptr)
			{
				GNoThrottleHooked = true;
				GEditor->ShouldDisableCPUThrottlingDelegates.Add(UEditorEngine::FShouldDisableCPUThrottling::CreateLambda(
					[]() { return FPlatformTime::Seconds() - GStreamingAt < 5.0; }));
				UE_LOG(LogRemoteHands, Log, TEXT("The editor will not throttle in the background while the robot's eyes stream."));
			}
			UEditorPerformanceSettings* Perf = GetMutableDefault<UEditorPerformanceSettings>();
			if (Perf != nullptr && Perf->bThrottleCPUWhenNotForeground)
			{
				Perf->bThrottleCPUWhenNotForeground = false;
				UE_LOG(LogRemoteHands, Log, TEXT("Turned off the editor's background CPU throttle while the robot's eyes stream."));
			}
		}
#endif
#if WITH_EDITOR
		if (bGot && !bGame && GEditor != nullptr && !GViewportsPaused)
		{
			GEditor->RedrawLevelEditingViewports(false);   // a viewport that is not "realtime" would not repaint by itself
		}
#endif
		return true;
	}

	void Register(ARemoteHandsReceiver* R)
	{
		GReceivers.AddUnique(R);
		if (!GTicker.IsValid())
		{
			GTicker = FTSTicker::GetCoreTicker().AddTicker(FTickerDelegate::CreateStatic(&Pump));
		}
	}

	void Unregister(ARemoteHandsReceiver* R)
	{
		GReceivers.Remove(R);
	}
}

void ARemoteHandsReceiver::PostRegisterAllComponents()
{
	Super::PostRegisterAllComponents();
	if (!IsTemplate())
	{
		RemoteHandsLink::Register(this);
	}
}

void ARemoteHandsReceiver::PostUnregisterAllComponents()
{
	RemoteHandsLink::Unregister(this);
	Super::PostUnregisterAllComponents();
}

void ARemoteHandsReceiver::BeginDestroy()
{
	RemoteHandsLink::Unregister(this);
	Super::BeginDestroy();
}

void ARemoteHandsReceiver::CaptureEyes()
{
	if (EyeCamera == nullptr || EyeCamera->GetRootComponent() == nullptr || GetWorld() == nullptr)
	{
		return;
	}
	const double T0 = FPlatformTime::Seconds();
	// A headset that told us where its wearer looks within the last second gets the head-following view.
	const bool bHead = T0 - RemoteHandsLink::GHeadAt < 1.0;
	const int32 W = bHead ? EyeVrWidth : EyeWidth, H = bHead ? EyeVrHeight : EyeHeight;
	const float Fov = bHead ? EyeVrFov : EyeFov;
	const float HeadYaw = RemoteHandsLink::GHeadYaw, HeadPitch = RemoteHandsLink::GHeadPitch;
	if (EyeTarget == nullptr || EyeTarget->SizeX != W || EyeTarget->SizeY != H)
	{
		EyeTarget = NewObject<UTextureRenderTarget2D>(this, NAME_None, RF_Transient);
		EyeTarget->RenderTargetFormat = RTF_RGBA8_SRGB;
		EyeTarget->InitAutoFormat(W, H);
		EyeTarget->UpdateResourceImmediate(true);
	}
	if (EyeCapture == nullptr || EyeCapture->GetAttachParent() != EyeCamera->GetRootComponent())
	{
		if (EyeCapture != nullptr)
		{
			EyeCapture->DestroyComponent();
		}
		EyeCapture = NewObject<USceneCaptureComponent2D>(EyeCamera, NAME_None, RF_Transient);
		EyeCapture->SetupAttachment(EyeCamera->GetRootComponent());
		EyeCapture->bCaptureEveryFrame = false;
		EyeCapture->bCaptureOnMovement = false;
		EyeCapture->bAlwaysPersistRenderingState = true;        // keeps exposure and lighting history between our captures
		EyeCapture->CaptureSource = ESceneCaptureSource::SCS_FinalColorLDR;
		EyeCapture->RegisterComponent();
	}
	EyeCapture->TextureTarget = EyeTarget;
	EyeCapture->FOVAngle = Fov;
	if (bHead)
	{
		// Level with the floor, turned from the way the robot faces by the wearer's head: left is positive for
		// the headset, right is positive for an Unreal yaw.
		EyeCapture->SetWorldRotation(FRotator(HeadPitch, GetActorRotation().Yaw - HeadYaw, 0.f));
	}
	else
	{
		EyeCapture->SetRelativeRotation(FRotator::ZeroRotator);      // the camera's own mount: looking down at the table
	}
	EyeCapture->CaptureScene();
	FTextureRenderTargetResource* Res = EyeTarget->GameThread_GetRenderTargetResource();
	TArray<FColor> Pixels;
	if (Res == nullptr || !Res->ReadPixels(Pixels) || Pixels.Num() != W * H)
	{
		return;
	}
	for (FColor& C : Pixels)
	{
		C.A = 255;
	}
	TArray64<uint8> Jpeg;
	if (!FImageUtils::CompressImage(Jpeg, TEXT("jpg"), FImageView(Pixels.GetData(), W, H), EyeJpegQuality))
	{
		return;
	}
	{
		FScopeLock Lock(&RemoteHandsLink::GEyeLock);
		RemoteHandsLink::GEyeJpeg.SetNumUninitialized(int32(Jpeg.Num()));
		FMemory::Memcpy(RemoteHandsLink::GEyeJpeg.GetData(), Jpeg.GetData(), Jpeg.Num());
		++RemoteHandsLink::GEyeSeq;
		RemoteHandsLink::GEyeTag = bHead ? FString::Printf(TEXT("%.2f,%.2f,%.1f,%.4f"), HeadYaw, HeadPitch, Fov, float(W) / float(H)) : FString(TEXT("flat"));
	}
	++EyeFrames;
	EyeFrameMs = float((FPlatformTime::Seconds() - T0) * 1000.0);
}

void ARemoteHandsReceiver::CapturePerson()
{
	const int32 W = 854, H = 480;
	if (PersonTarget == nullptr)
	{
		PersonTarget = NewObject<UTextureRenderTarget2D>(this, NAME_None, RF_Transient);
		PersonTarget->RenderTargetFormat = RTF_RGBA8_SRGB;
		PersonTarget->InitAutoFormat(W, H);
		PersonTarget->UpdateResourceImmediate(true);
	}
	if (PersonCapture == nullptr || PersonCapture->GetAttachParent() != PersonCamera->GetRootComponent())
	{
		if (PersonCapture != nullptr)
		{
			PersonCapture->DestroyComponent();
		}
		PersonCapture = NewObject<USceneCaptureComponent2D>(PersonCamera, NAME_None, RF_Transient);
		PersonCapture->SetupAttachment(PersonCamera->GetRootComponent());
		PersonCapture->bCaptureEveryFrame = false;
		PersonCapture->bCaptureOnMovement = false;
		PersonCapture->bAlwaysPersistRenderingState = true;
		PersonCapture->CaptureSource = ESceneCaptureSource::SCS_FinalColorLDR;
		PersonCapture->RegisterComponent();
	}
	PersonCapture->TextureTarget = PersonTarget;
	PersonCapture->FOVAngle = PersonFov;
	PersonCapture->CaptureScene();
	FTextureRenderTargetResource* Res = PersonTarget->GameThread_GetRenderTargetResource();
	TArray<FColor> Pixels;
	if (Res == nullptr || !Res->ReadPixels(Pixels) || Pixels.Num() != W * H)
	{
		return;
	}
	for (FColor& C : Pixels)
	{
		C.A = 255;
	}
	TArray64<uint8> Jpeg;
	if (!FImageUtils::CompressImage(Jpeg, TEXT("jpg"), FImageView(Pixels.GetData(), W, H), 70))
	{
		return;
	}
	FScopeLock Lock(&RemoteHandsLink::GEyeLock);
	RemoteHandsLink::GPersonJpeg.SetNumUninitialized(int32(Jpeg.Num()));
	FMemory::Memcpy(RemoteHandsLink::GPersonJpeg.GetData(), Jpeg.GetData(), Jpeg.Num());
	++RemoteHandsLink::GPersonSeq;
}

void ARemoteHandsReceiver::NoteLink(double Now, bool bGotPacket)
{
	if (bGotPacket)
	{
		LastPacketAt = Now;
	}
	bConnected = LastPacketAt > 0.0 && Now - LastPacketAt < TimeoutSeconds;
}

void ARemoteHandsReceiver::Tick(float DeltaSeconds)
{
	Super::Tick(DeltaSeconds);
	if (bDrawDebug && PacketsReceived > 0 && GetWorld() != nullptr && !IsTemplate())
	{
		DrawDebug();
	}
}

bool ARemoteHandsReceiver::ApplyPacket(const FString& Json)
{
	TSharedPtr<FJsonObject> Obj;
	if (!FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(Json), Obj) || !Obj.IsValid() ||
		Obj->GetStringField(TEXT("type")) != TEXT("robot_state"))
	{
		return false;
	}
	FRemoteHandsState S;
	S.Seq = int32(Obj->GetNumberField(TEXT("seq")));
	S.Time = float(Obj->GetNumberField(TEXT("t")));
	const TSharedPtr<FJsonObject>* Sub = nullptr;
	const TSharedPtr<FJsonObject>* Ext = nullptr;
	const bool bExt = Obj->TryGetObjectField(TEXT("ext"), Ext);
	if (bExt)
	{
		(*Ext)->TryGetStringField(TEXT("mode"), S.Mode);
		(*Ext)->TryGetStringField(TEXT("motion_source"), S.MotionSource);
		// Where the robot stands. Done first: every robot-frame position below goes through this actor's transform.
		FTransform Pelvis;
		if ((*Ext)->TryGetObjectField(TEXT("pelvis"), Sub) && ReadPose(*Sub, Pelvis))
		{
			S.bHasBase = true;
			if (bFollowBase)
			{
				const FTransform World = Pelvis * SimWorldOrigin;
				SetActorLocationAndRotation(World.GetLocation(), World.GetRotation(), false, nullptr, ETeleportType::TeleportPhysics);
				if (RobotActor != nullptr)
				{
					RobotActor->SetActorLocationAndRotation(World.GetLocation(), World.GetRotation(), false, nullptr, ETeleportType::TeleportPhysics);
				}
			}
		}
		if ((*Ext)->TryGetObjectField(TEXT("joints"), Sub))        // legs and waist
		{
			for (const auto& KV : (*Sub)->Values)
			{
				S.Joints.Add(FName(*KV.Key), float(KV.Value->AsNumber()));
			}
		}
	}
	if (Obj->TryGetObjectField(TEXT("joints"), Sub))
	{
		for (const auto& KV : (*Sub)->Values)
		{
			S.Joints.Add(FName(*KV.Key), float(KV.Value->AsNumber()));
		}
	}
	const TArray<TSharedPtr<FJsonValue>>* Arr = nullptr;
	FVector V;
	if (Obj->TryGetObjectField(TEXT("hands"), Sub))
	{
		if ((*Sub)->TryGetArrayField(TEXT("left"), Arr) && ReadVec(Arr, V))
		{
			S.LeftHandWorld = RobotToWorld(V);
		}
		if ((*Sub)->TryGetArrayField(TEXT("right"), Arr) && ReadVec(Arr, V))
		{
			S.RightHandWorld = RobotToWorld(V);
		}
	}
	if (Obj->TryGetObjectField(TEXT("closed"), Sub))
	{
		(*Sub)->TryGetBoolField(TEXT("left"), S.bLeftClosed);
		(*Sub)->TryGetBoolField(TEXT("right"), S.bRightClosed);
	}
	if (Obj->TryGetArrayField(TEXT("items"), Arr))
	{
		for (const TSharedPtr<FJsonValue>& Value : *Arr)
		{
			const TSharedPtr<FJsonObject>* ItemObj = nullptr;
			const TArray<TSharedPtr<FJsonValue>>* Pos = nullptr;
			if (!Value->TryGetObject(ItemObj) || !(*ItemObj)->TryGetArrayField(TEXT("pos"), Pos) || !ReadVec(Pos, V))
			{
				continue;
			}
			FRemoteHandsItem Item;
			Item.Name = (*ItemObj)->GetStringField(TEXT("name"));
			Item.WorldLocation = GetActorTransform().TransformPosition(RobotMetresToLocalCm(V) + ItemOffset);
			(*ItemObj)->TryGetStringField(TEXT("held_by"), Item.HeldBy);     // null -> stays empty
			const TSharedPtr<FJsonObject>* ExtItems = nullptr;
			const TSharedPtr<FJsonObject>* ItemPose = nullptr;
			FTransform Pose;
			if (bExt && (*Ext)->TryGetObjectField(TEXT("items"), ExtItems) && (*ExtItems)->TryGetObjectField(Item.Name, ItemPose) &&
				ReadPose(*ItemPose, Pose))
			{
				const FTransform World = Pose * SimWorldOrigin;         // the full measured pose, in the room
				Item.WorldLocation = World.GetLocation() + World.GetRotation().RotateVector(ItemOffset);
				Item.WorldRotation = World.GetRotation();
				Item.bHasRotation = true;
			}
			S.Items.Add(MoveTemp(Item));
		}
	}
	if (Obj->TryGetObjectField(TEXT("task"), Sub))
	{
		(*Sub)->TryGetStringField(TEXT("item"), S.TaskItem);
		(*Sub)->TryGetStringField(TEXT("state"), S.TaskState);
	}
	if (Obj->TryGetArrayField(TEXT("tray"), Arr) && ReadVec(Arr, V))
	{
		S.TrayWorld = RobotToWorld(V);
	}
	double TableZ = 0.0;
	if (Obj->TryGetNumberField(TEXT("table_z"), TableZ))
	{
		S.TableWorldZ = float(RobotToWorld(FVector(0.3, 0.0, TableZ)).Z);
	}

	const TSharedPtr<FJsonObject>* Door = nullptr;
	if (bExt && (*Ext)->TryGetObjectField(TEXT("door"), Door))
	{
		double Angle = 0.0;
		if ((*Door)->TryGetNumberField(TEXT("angle"), Angle))
		{
			S.DoorAngle = float(Angle);
			S.bHasDoor = true;
			FTransform Hinge;
			const TSharedPtr<FJsonObject>* HingeObj = nullptr;
			if (DoorActor != nullptr && (*Door)->TryGetObjectField(TEXT("hinge"), HingeObj) && ReadPose(*HingeObj, Hinge))
			{
				FVector Axis(0, 0, 1);
				if ((*HingeObj)->TryGetArrayField(TEXT("axis"), Arr))
				{
					ReadVec(Arr, Axis);
				}
				// +a about (x, y, z) in the sender's frame is +a about (-x, y, -z) here (y is mirrored).
				const FQuat Swing(FVector(-Axis.X, Axis.Y, -Axis.Z).GetSafeNormal(UE_SMALL_NUMBER, FVector::DownVector), Angle);
				const FTransform World = FTransform(Swing * Hinge.GetRotation(), Hinge.GetLocation()) * SimWorldOrigin;
				DoorActor->SetActorLocationAndRotation(World.GetLocation(), World.GetRotation(), false, nullptr, ETeleportType::TeleportPhysics);
			}
		}
	}

	// The demo scene: where the wheelchair is and which pose of the person to show.
	const TSharedPtr<FJsonObject>* Scene = nullptr;
	if (bExt && (*Ext)->TryGetObjectField(TEXT("scene"), Scene))
	{
		const double Now = FPlatformTime::Seconds();
		FTransform ChairWorld = ChairActor != nullptr ? ChairActor->GetActorTransform() : FTransform::Identity;
		const TArray<TSharedPtr<FJsonValue>>* C = nullptr;
		if ((*Scene)->TryGetArrayField(TEXT("chair"), C) && C->Num() >= 3)
		{
			// room frame: x forward, y left, heading anticlockwise (radians) -> Unreal: y and the heading mirrored
			const FTransform Local(FRotator(0.0, -FMath::RadiansToDegrees((*C)[2]->AsNumber()) + ChairYawOffset, 0.0),
				FVector(100.0 * (*C)[0]->AsNumber(), -100.0 * (*C)[1]->AsNumber(), 0.0));
			ChairWorld = Local * SimWorldOrigin;
			if (ChairActor != nullptr)
			{
				ChairActor->SetActorLocationAndRotation(ChairWorld.GetLocation(), ChairWorld.GetRotation(), false, nullptr, ETeleportType::TeleportPhysics);
			}
		}
		FString Man;
		(*Scene)->TryGetStringField(TEXT("man"), Man);
		double Cough = 0.0;
		if ((*Scene)->TryGetNumberField(TEXT("cough"), Cough) && int32(Cough) != LastCough)
		{
			if (LastCough >= 0)
			{
				CoughAt = Now;
			}
			LastCough = int32(Cough);
		}
		if (PersonCamera != nullptr)        // his own view: from his eyes, in the chair or in the bed
		{
			const FTransform Eye = Man == BedPoseName ? PersonEyeInBed * BedPoseTransform : PersonEyeInChair * ChairWorld;
			PersonCamera->SetActorLocationAndRotation(Eye.GetLocation(), Eye.GetRotation(), false, nullptr, ETeleportType::TeleportPhysics);
		}
		// a cough rocks him in the chair for a second and a half
		const double T = Now - CoughAt;
		const double Rock = T < 1.5 ? 4.0 * FMath::Sin(T * 21.0) * FMath::Exp(-2.0 * T) : 0.0;
		for (const TPair<FString, TObjectPtr<AActor>>& P : ManPoses)
		{
			if (P.Value == nullptr)
			{
				continue;
			}
			if (P.Key != Man)
			{
				P.Value->SetActorLocation(FVector(0.0, 0.0, -100000.0), false, nullptr, ETeleportType::TeleportPhysics);
				continue;
			}
			const FTransform W = P.Key == BedPoseName ? BedPoseTransform : FTransform(FRotator(0.0, 0.0, Rock)) * ChairWorld;
			P.Value->SetActorLocationAndRotation(W.GetLocation(), W.GetRotation(), false, nullptr, ETeleportType::TeleportPhysics);
		}
	}

	const bool bTaskChanged = S.TaskItem != State.TaskItem || S.TaskState != State.TaskState;
	State = MoveTemp(S);
	++PacketsReceived;
	ApplyJoints();
	ApplyItems();
	OnState.Broadcast(State);
	if (bTaskChanged)
	{
		OnTaskStateChanged.Broadcast(State.TaskItem, State.TaskState);
	}
	return true;
}

void ARemoteHandsReceiver::ApplyJoints()
{
	if (RobotActor == nullptr || Joints.Num() == 0)
	{
		return;
	}
	UPoseableMeshComponent* Poseable = nullptr;
	bool bBonesTouched = false;
	for (const TPair<FName, FRemoteHandsJointBinding>& KV : Joints)
	{
		const float* Angle = State.Joints.Find(KV.Key);
		const FRemoteHandsJointBinding& B = KV.Value;
		if (Angle == nullptr || B.Target.IsNone())
		{
			continue;
		}
		const FQuat Turn(B.Axis.GetSafeNormal(UE_SMALL_NUMBER, FVector::UpVector), *Angle * B.Scale + FMath::DegreesToRadians(B.OffsetDegrees));
		const FQuat Rotation = B.Rest * Turn;       // the joint turns in the part's own frame, after its rest pose
		if (B.bIsBone)
		{
			Poseable = Poseable != nullptr ? Poseable : FindPoseable();
			const int32 Bone = Poseable != nullptr ? Poseable->GetBoneIndex(B.Target) : INDEX_NONE;
			if (Bone != INDEX_NONE && Poseable->BoneSpaceTransforms.IsValidIndex(Bone))
			{
				Poseable->BoneSpaceTransforms[Bone].SetRotation(Rotation);
				bBonesTouched = true;
			}
		}
		else if (USceneComponent* Part = FindPart(RobotActor, B.Target.ToString()))
		{
			Part->SetRelativeRotation(Rotation);
		}
	}
	if (bBonesTouched)
	{
		Poseable->MarkRefreshTransformDirty();
	}
}

void ARemoteHandsReceiver::ApplyItems()
{
	for (const FRemoteHandsItem& Item : State.Items)
	{
		if (const TObjectPtr<AActor>* Actor = Items.Find(Item.Name))
		{
			if (*Actor != nullptr)
			{
				if (Item.bHasRotation)
				{
					(*Actor)->SetActorLocationAndRotation(Item.WorldLocation, Item.WorldRotation, false, nullptr, ETeleportType::TeleportPhysics);
				}
				else
				{
					(*Actor)->SetActorLocation(Item.WorldLocation, false, nullptr, ETeleportType::TeleportPhysics);
				}
			}
		}
	}
}

void ARemoteHandsReceiver::DrawDebug() const
{
	const UWorld* World = GetWorld();
	const FColor Link = bConnected ? FColor::Green : FColor(120, 120, 120);
	DrawDebugSphere(World, State.LeftHandWorld, 3.f, 12, State.bLeftClosed ? FColor::Red : Link, false, -1.f, 0, 0.3f);
	DrawDebugSphere(World, State.RightHandWorld, 3.f, 12, State.bRightClosed ? FColor::Red : Link, false, -1.f, 0, 0.3f);
	DrawDebugString(World, State.LeftHandWorld + FVector(0, 0, 6), TEXT("L hand"), nullptr, Link, 0.f);
	DrawDebugString(World, State.RightHandWorld + FVector(0, 0, 6), TEXT("R hand"), nullptr, Link, 0.f);
	for (const FRemoteHandsItem& Item : State.Items)
	{
		const bool bWanted = Item.Name == State.TaskItem;
		DrawDebugBox(World, Item.WorldLocation, FVector(2.25f), GetActorQuat(), bWanted ? FColor::Yellow : FColor::Cyan, false, -1.f, 0, 0.2f);
		DrawDebugString(World, Item.WorldLocation + FVector(0, 0, 5),
			Item.HeldBy.IsEmpty() ? Item.Name : Item.Name + TEXT(" (") + Item.HeldBy + TEXT(" hand)"), nullptr,
			bWanted ? FColor::Yellow : FColor::Cyan, 0.f);
	}
	DrawDebugSphere(World, State.TrayWorld, 2.f, 8, FColor::Orange, false, -1.f, 0, 0.2f);
	DrawDebugString(World, State.TrayWorld + FVector(0, 0, 4), TEXT("put-down point"), nullptr, FColor::Orange, 0.f);
	const FVector TableCentre(RobotToWorld(FVector(0.39, 0.0, 0.0)).X, RobotToWorld(FVector(0.39, 0.0, 0.0)).Y, State.TableWorldZ);
	DrawDebugBox(World, TableCentre, FVector(25.f, 45.f, 0.1f), GetActorQuat(), FColor(160, 110, 60), false, -1.f, 0, 0.2f);
	if (!State.TaskItem.IsEmpty())
	{
		DrawDebugString(World, GetActorLocation() + FVector(0, 0, 60), FString::Printf(TEXT("%s: %s"), *State.TaskItem, *State.TaskState),
			nullptr, FColor::Yellow, 0.f);
	}
}

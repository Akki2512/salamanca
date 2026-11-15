// Remote Hands Link: the Unreal end of the robot node's --unreal-udp stream (remote_hands/unreal_stream.py).
//
// Unreal only draws. Every packet (30 a second, JSON over UDP) carries the G1's 28 arm and hand joint angles and
// the position of every item; this actor applies them. Nothing is simulated here: an item in the robot's hand
// follows the hand because the robot node says where it is.
//
// Setup in a level:
//   1. Place one RemoteHandsReceiver AT THE ROBOT'S PELVIS (waist joint), its X axis pointing where the robot
//      faces. This actor's transform is the robot frame that all positions in the packets are relative to.
//   2. Set RobotActor to the G1. Press "Auto Bind Joints" (Details panel). It looks, on that actor, for a
//      scene component or a bone of a Poseable Mesh named after each joint ("left_elbow_joint") or its link
//      ("left_elbow_link"), and remembers the pose it finds as the zero pose.
//   3. Fill Items: item name ("pill bottle") -> the actor to move.
//   4. It also works in the editor viewport without pressing Play (bRunInEditor).
//
// One UDP listener is shared by every receiver in the process and runs on the engine's core ticker, so packets
// are read every frame whether or not a level is being ticked. While a Play / Simulate session exists the
// packets go to the receivers in that session; otherwise to the ones in the level being edited.
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "RemoteHandsReceiver.generated.h"

class FSocket;
class USceneComponent;
class UPoseableMeshComponent;

/** How one joint of the packet drives one part of the robot. */
USTRUCT(BlueprintType)
struct FRemoteHandsJointBinding
{
	GENERATED_BODY()

	/** Scene component or bone to rotate. Filled by Auto Bind Joints. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Remote Hands")
	FName Target;

	/** Rotation axis in the part's own frame, in UNREAL axes (X forward, Y right, Z up). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Remote Hands")
	FVector Axis = FVector::UpVector;

	/** Multiplies the angle. Set to -1 if this joint turns the wrong way on your rig. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Remote Hands")
	float Scale = 1.f;

	/** Added to the angle, degrees. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Remote Hands")
	float OffsetDegrees = 0.f;

	/** The part's rotation relative to its parent at joint angle zero. Captured by Auto Bind Joints. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Remote Hands")
	FQuat Rest = FQuat::Identity;

	/** True when Target is a bone of the robot's Poseable Mesh, false when it is a scene component. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Remote Hands")
	bool bIsBone = false;
};

/** One item on the table as the robot node sees it. Position already converted to world space. */
USTRUCT(BlueprintType)
struct FRemoteHandsItem
{
	GENERATED_BODY()

	UPROPERTY(BlueprintReadOnly, Category = "Remote Hands")
	FString Name;

	UPROPERTY(BlueprintReadOnly, Category = "Remote Hands")
	FVector WorldLocation = FVector::ZeroVector;

	/** "left", "right" or empty. */
	UPROPERTY(BlueprintReadOnly, Category = "Remote Hands")
	FString HeldBy;

	/** World rotation, when the sender gives full item poses (ext.items); otherwise bHasRotation is false. */
	UPROPERTY(BlueprintReadOnly, Category = "Remote Hands")
	FQuat WorldRotation = FQuat::Identity;

	UPROPERTY(BlueprintReadOnly, Category = "Remote Hands")
	bool bHasRotation = false;
};

/** The newest packet, decoded. */
USTRUCT(BlueprintType)
struct FRemoteHandsState
{
	GENERATED_BODY()

	UPROPERTY(BlueprintReadOnly, Category = "Remote Hands")
	int32 Seq = 0;

	/** Robot time, seconds. */
	UPROPERTY(BlueprintReadOnly, Category = "Remote Hands")
	float Time = 0.f;

	/** Joint name -> angle in radians, as sent. */
	UPROPERTY(BlueprintReadOnly, Category = "Remote Hands")
	TMap<FName, float> Joints;

	UPROPERTY(BlueprintReadOnly, Category = "Remote Hands")
	FVector LeftHandWorld = FVector::ZeroVector;

	UPROPERTY(BlueprintReadOnly, Category = "Remote Hands")
	FVector RightHandWorld = FVector::ZeroVector;

	UPROPERTY(BlueprintReadOnly, Category = "Remote Hands")
	bool bLeftClosed = false;

	UPROPERTY(BlueprintReadOnly, Category = "Remote Hands")
	bool bRightClosed = false;

	UPROPERTY(BlueprintReadOnly, Category = "Remote Hands")
	TArray<FRemoteHandsItem> Items;

	/** The item being fetched ("pill bottle"), empty when there is no task. */
	UPROPERTY(BlueprintReadOnly, Category = "Remote Hands")
	FString TaskItem;

	/** free, approach, descend, closing, holding, releasing, done; empty when there is no task. */
	UPROPERTY(BlueprintReadOnly, Category = "Remote Hands")
	FString TaskState;

	/** Where the robot puts things down for the person. */
	UPROPERTY(BlueprintReadOnly, Category = "Remote Hands")
	FVector TrayWorld = FVector::ZeroVector;

	/** Height of the table top, world Z. */
	UPROPERTY(BlueprintReadOnly, Category = "Remote Hands")
	float TableWorldZ = 0.f;

	/** True when the packet carried the robot's own position in the room (ext.pelvis): the robot walks. */
	UPROPERTY(BlueprintReadOnly, Category = "Remote Hands")
	bool bHasBase = false;

	/** Door hinge angle in radians, 0 = closed (ext.door); bHasDoor false when not sent. */
	UPROPERTY(BlueprintReadOnly, Category = "Remote Hands")
	float DoorAngle = 0.f;

	UPROPERTY(BlueprintReadOnly, Category = "Remote Hands")
	bool bHasDoor = false;

	/** Sender's mode and where the motion comes from ("live suit", "scripted", ...), when sent. */
	UPROPERTY(BlueprintReadOnly, Category = "Remote Hands")
	FString Mode;

	UPROPERTY(BlueprintReadOnly, Category = "Remote Hands")
	FString MotionSource;
};

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FRemoteHandsStateEvent, const FRemoteHandsState&, State);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FRemoteHandsTaskEvent, const FString&, Item, const FString&, NewState);

UCLASS(Blueprintable, meta = (DisplayName = "Remote Hands Receiver"))
class REMOTEHANDSLINK_API ARemoteHandsReceiver : public AActor
{
	GENERATED_BODY()

public:
	ARemoteHandsReceiver();

	/** UDP port the robot node sends to (--unreal-udp HOST:PORT). */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Remote Hands|Link")
	int32 Port = 7777;

	/** More ports to listen on at the same time, if any. The newest packet on any port wins. */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Remote Hands|Link")
	TArray<int32> ExtraPorts;

	/** Also listen and pose while editing, without pressing Play. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Remote Hands|Link")
	bool bRunInEditor = true;

	/** No packet for this long = link lost (bConnected goes false; the pose is kept). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Remote Hands|Link")
	float TimeoutSeconds = 0.5f;

	/** The G1. Its parts are rotated by the joint angles. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Remote Hands|Robot")
	TObjectPtr<AActor> RobotActor;

	/** When the packet says where the robot is in the room (ext.pelvis), move the robot and this actor there. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Remote Hands|Robot")
	bool bFollowBase = true;

	/** Where the sender's world origin (floor level, x forward, y left, z up) is in this level. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Remote Hands|Robot")
	FTransform SimWorldOrigin = FTransform::Identity;

	/** The door to swing from ext.door. Its pivot must be on the hinge line; its closed pose is taken from the packet's hinge. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Remote Hands|Room")
	TObjectPtr<AActor> DoorActor;

	/** The wheelchair. It is put where ext.scene.chair says ([x, y, heading] in the sender's room frame). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Remote Hands|Scene")
	TObjectPtr<AActor> ChairActor;

	/** Added to the chair's heading, degrees: the chair model faces its own +Y, so -90. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Remote Hands|Scene")
	float ChairYawOffset = -90.f;

	/** The person, once per pose (ext.scene.man names the one to show: "idle", "hand_up", "bed"). The pose shown
	 *  rides the chair; the others are parked out of sight under the floor. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Remote Hands|Scene")
	TMap<FString, TObjectPtr<AActor>> ManPoses;

	/** A camera for the person's own view, served at http://127.0.0.1:<EyeHttpPort>/person.jpg. It is put at his
	 *  eyes: PersonEyeInChair (in the chair's axes) while he sits, PersonEyeInBed (in BedPoseTransform's) in bed. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Remote Hands|Scene")
	TObjectPtr<AActor> PersonCamera;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Remote Hands|Scene")
	FTransform PersonEyeInChair = FTransform(FRotator(-8.0, 90.0, 0.0), FVector(0.0, 24.0, 141.0));

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Remote Hands|Scene")
	FTransform PersonEyeInBed = FTransform(FRotator(28.0, 90.0, 0.0), FVector(0.0, -66.0, 26.0));

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Remote Hands|Scene")
	float PersonFps = 12.f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Remote Hands|Scene")
	float PersonFov = 85.f;

	/** The pose that does not ride the chair (lying in bed), and where it lies. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Remote Hands|Scene")
	FString BedPoseName = TEXT("bed");

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Remote Hands|Scene")
	FTransform BedPoseTransform = FTransform::Identity;

	/** Joint name in the packet -> what it turns. Filled by Auto Bind Joints; edit Scale to flip a joint. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Remote Hands|Robot")
	TMap<FName, FRemoteHandsJointBinding> Joints;

	/** Item name in the packet ("pill bottle") -> the actor to move. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Remote Hands|Items")
	TMap<FString, TObjectPtr<AActor>> Items;

	/** Added to an item's position, in the robot frame's Unreal axes, centimetres (for meshes whose origin is not at their centre). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Remote Hands|Items")
	FVector ItemOffset = FVector::ZeroVector;

	/** The camera whose picture is served for the worker's headset (the robot's eyes). Empty = no video. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Remote Hands|Eyes")
	TObjectPtr<AActor> EyeCamera;

	/** Serve EyeCamera's picture as JPEG at http://127.0.0.1:<EyeHttpPort>/eyes.jpg (newest frame; the viewer page polls it). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Remote Hands|Eyes")
	bool bStreamEyes = true;

	/** While the eye picture is streaming, stop the editor's own level viewports from redrawing. Otherwise the
	 *  scene is rendered twice a frame (the viewport and the robot's eyes) and each eye frame waits for the viewport's. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Remote Hands|Eyes")
	bool bPauseEditorViewports = true;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Remote Hands|Eyes")
	int32 EyeHttpPort = 8011;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Remote Hands|Eyes", meta = (ClampMin = 160, ClampMax = 2560))
	int32 EyeWidth = 1280;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Remote Hands|Eyes", meta = (ClampMin = 90, ClampMax = 1440))
	int32 EyeHeight = 720;

	/** Frames a second rendered and encoded. Each one costs a second render of the scene. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Remote Hands|Eyes", meta = (ClampMin = 1, ClampMax = 60))
	float EyeFps = 30.f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Remote Hands|Eyes", meta = (ClampMin = 10, ClampMax = 100))
	int32 EyeJpegQuality = 70;

	/** Field of view of the served picture, degrees (horizontal). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Remote Hands|Eyes", meta = (ClampMin = 30, ClampMax = 140))
	float EyeFov = 90.f;

	/** Headset view: when the viewer sends where the wearer's head points (eyes.jpg?vr=1&y=..&p=..), the camera
	 *  turns to look that way (relative to the way the robot faces) and this wider, taller picture is rendered. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Remote Hands|Eyes", meta = (ClampMin = 160, ClampMax = 2560))
	int32 EyeVrWidth = 1280;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Remote Hands|Eyes", meta = (ClampMin = 90, ClampMax = 2048))
	int32 EyeVrHeight = 1024;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Remote Hands|Eyes", meta = (ClampMin = 60, ClampMax = 140))
	float EyeVrFov = 110.f;

	/** Frames served since the level loaded, and how long the last one took to render and encode (ms). */
	UPROPERTY(BlueprintReadOnly, Category = "Remote Hands|Eyes")
	int32 EyeFrames = 0;

	UPROPERTY(BlueprintReadOnly, Category = "Remote Hands|Eyes")
	float EyeFrameMs = 0.f;

	/** Render EyeCamera's view now and make it the frame the HTTP route serves. Called by the shared ticker. */
	void CaptureEyes();
	void CapturePerson();

	/** Draw the hands, items, tray and table height as debug shapes: check alignment before the robot is rigged. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Remote Hands|Debug")
	bool bDrawDebug = true;

	UPROPERTY(BlueprintReadOnly, Category = "Remote Hands")
	FRemoteHandsState State;

	UPROPERTY(BlueprintReadOnly, Category = "Remote Hands")
	bool bConnected = false;

	/** Packets received since the socket opened. */
	UPROPERTY(BlueprintReadOnly, Category = "Remote Hands")
	int32 PacketsReceived = 0;

	/** Every time a new packet has been applied. */
	UPROPERTY(BlueprintAssignable, Category = "Remote Hands")
	FRemoteHandsStateEvent OnState;

	/** When the fetch moves on: e.g. ("pill bottle", "holding"), ("pill bottle", "done"). */
	UPROPERTY(BlueprintAssignable, Category = "Remote Hands")
	FRemoteHandsTaskEvent OnTaskStateChanged;

	/** Find the robot's parts by joint / link name and record the current pose as zero. */
	UFUNCTION(CallInEditor, BlueprintCallable, Category = "Remote Hands|Robot")
	void AutoBindJoints();

	/** Sender's world frame (metres, x forward, y left, z up) -> world space here, through SimWorldOrigin. */
	UFUNCTION(BlueprintPure, Category = "Remote Hands")
	FVector SimToWorld(const FVector& SimMetres) const;

	/** Robot frame (metres, x forward, y left, z up) -> world space. */
	UFUNCTION(BlueprintPure, Category = "Remote Hands")
	FVector RobotToWorld(const FVector& RobotMetres) const;

	/** Decode one packet and apply it. Returns false if the text is not a robot_state packet. */
	UFUNCTION(BlueprintCallable, Category = "Remote Hands")
	bool ApplyPacket(const FString& Json);

	virtual void Tick(float DeltaSeconds) override;
	virtual bool ShouldTickIfViewportsOnly() const override { return bRunInEditor; }
	virtual void PostRegisterAllComponents() override;
	virtual void PostUnregisterAllComponents() override;
	virtual void BeginDestroy() override;

	/** Called by the shared listener when no packet has been applied for a while / one just was. */
	void NoteLink(double Now, bool bGotPacket);

private:
	int32 LastCough = -1;
	double CoughAt = -100.0;
	void ApplyJoints();
	void ApplyItems();
	void DrawDebug() const;
	UPoseableMeshComponent* FindPoseable() const;

	double LastPacketAt = 0.0;

	UPROPERTY(Transient)
	TObjectPtr<class USceneCaptureComponent2D> EyeCapture;

	UPROPERTY(Transient)
	TObjectPtr<class UTextureRenderTarget2D> EyeTarget;

	UPROPERTY(Transient)
	TObjectPtr<class USceneCaptureComponent2D> PersonCapture;

	UPROPERTY(Transient)
	TObjectPtr<class UTextureRenderTarget2D> PersonTarget;
};
